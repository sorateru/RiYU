mod actions;
mod columnar;
mod physical;
mod replay;
mod shanten;
mod source;
mod teachers;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONTRACT: &str = "riyu-data-lake-pilot/1";

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.first().is_some_and(|s| s == "--help") {
        eprintln!(
            "riyu-data-lake DB OUTPUT --authorization-reference REF --limit N [--offset N] [--plan FILE] [--workers N] [--shard-size N] [--format parquet|ipc|jsonl] [--profile A|B|C|D] [--checkpoint 16|32|64] [--batch N] [--row-group N] [--zstd 1|3|6] [--no-shanten-cache] [--no-checkpoints] [--normalize-only] [--measure-restore]"
        );
        ensure!(
            args.first().is_some_and(|s| s == "--help"),
            "database and output required"
        );
        return Ok(());
    }
    let db = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let mut options = BTreeMap::new();
    let mut flags = HashSet::new();
    let mut tokens = args[2..].iter();
    while let Some(flag) = tokens.next() {
        match flag.as_str() {
            "--authorization-reference"
            | "--limit"
            | "--offset"
            | "--workers"
            | "--shard-size"
            | "--format"
            | "--profile"
            | "--checkpoint"
            | "--batch"
            | "--row-group"
            | "--zstd"
            | "--plan" => {
                let value = tokens
                    .next()
                    .with_context(|| format!("missing value for {flag}"))?;
                ensure!(
                    !value.starts_with("--")
                        && options.insert(flag.as_str(), value.as_str()).is_none(),
                    "invalid repeated or empty option {flag}"
                );
            }
            "--no-shanten-cache"
            | "--no-checkpoints"
            | "--normalize-only"
            | "--measure-restore"
            | "--stop-on-replay-error" => {
                ensure!(flags.insert(flag.as_str()), "repeated flag {flag}")
            }
            _ => bail!("unknown option {flag}"),
        }
    }
    let number = |key: &str, default: usize| -> Result<usize> {
        Ok(options
            .get(key)
            .map(|x| x.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let rights = options
        .get("--authorization-reference")
        .filter(|s| !s.trim().is_empty())
        .context("local authorization reference is required")?;
    let limit = number("--limit", 0)?;
    let offset = number("--offset", 0)?;
    let shard_size = number("--shard-size", 16)?;
    let workers = number("--workers", 4)?;
    ensure!(
        (1..=10_000).contains(&limit)
            && (1..=32).contains(&workers)
            && (1..=32).contains(&shard_size),
        "trial limit 1..10000, workers and shard size 1..32"
    );
    offset.checked_add(limit).context("offset overflow")?;
    let config = columnar::Config {
        format: options.get("--format").unwrap_or(&"parquet").to_string(),
        profile: options.get("--profile").unwrap_or(&"D").to_string(),
        checkpoint: number("--checkpoint", 32)? as u64,
        batch: number("--batch", 4096)?,
        row_group: number("--row-group", 8192)?,
        zstd: number("--zstd", 3)?.try_into()?,
    };
    ensure!(
        matches!(config.format.as_str(), "jsonl" | "ipc" | "parquet")
            && matches!(config.profile.as_str(), "A" | "B" | "C" | "D")
            && matches!(config.checkpoint, 16 | 32 | 64)
            && matches!(config.zstd, 1 | 3 | 6)
            && (1..=65_536).contains(&config.batch)
            && (1..=524_288).contains(&config.row_group),
        "invalid columnar output settings"
    );
    ensure!(
        config.format == "jsonl"
            || !flags.contains("--no-checkpoints")
            || !matches!(config.profile.as_str(), "C" | "D"),
        "C/D require checkpoints"
    );
    let planned = options.get("--plan").map(fs::read_to_string).transpose()?;
    let locators = planned.as_ref().map(|p| p.lines().collect::<Vec<_>>());
    if let Some(ids) = &locators {
        ensure!(
            offset + limit <= ids.len()
                && ids.iter().collect::<HashSet<_>>().len() == ids.len()
                && ids.iter().all(|id| !id.is_empty()),
            "invalid source selection plan"
        );
    }

    let start = Instant::now();
    let mut container_hasher = Sha256::new();
    let db_size = db.metadata()?.len();
    std::io::copy(&mut File::open(db)?, &mut container_hasher)?;
    let container_sha = format!("{:x}", container_hasher.finalize());
    let executable_sha = hash_file(&std::env::current_exe()?)?;
    let source_hash_us = teachers::elapsed(start);
    let source = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut stmt =
        source.prepare("SELECT log_id, log_content FROM logs ORDER BY log_id LIMIT ? OFFSET ?")?;
    let mut lookup = source.prepare("SELECT log_content FROM logs WHERE log_id = ?")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    create_private_dir(output)?;
    let build = json!({"contract": CONTRACT, "container_sha256": container_sha, "container_bytes": db_size,
        "executable_sha256": executable_sha, "authorization_reference": rights,
        "cache": !flags.contains("--no-shanten-cache"), "checkpoints": !flags.contains("--no-checkpoints"),
        "measure_restore": flags.contains("--measure-restore"),
        "normalize_only": flags.contains("--normalize-only"), "offset": offset, "limit": limit,
        "shard_size": shard_size, "redistribution_allowed": false,
        "format":config.format, "profile":config.profile, "checkpoint":config.checkpoint,
        "batch":config.batch, "row_group":config.row_group, "zstd":config.zstd,
        "plan_sha256":planned.as_ref().map(|s| source::digest(s.as_bytes()))});
    let mut seen = HashSet::new();
    let mut action_registry = HashMap::new();
    let mut manifests = Vec::new();
    for relative in (0..limit).step_by(shard_size) {
        let offset = offset + relative;
        let length = shard_size.min(limit - relative);
        let path = output.join(if config.format == "jsonl" {
            format!("shard-{offset:08}-{length:04}.jsonl")
        } else {
            format!("shard-{offset:08}-{length:04}")
        });
        let header = json!({"type":"header", "build":build, "offset":offset, "requested":length});
        let start = Instant::now();
        let footer = if path.exists() {
            if config.format == "jsonl" {
                verify_shard(&path, &header, &mut seen)?
            } else {
                columnar::verify(&path, &header, &mut seen, &config.format)?
            }
        } else {
            let records = if let Some(ids) = &locators {
                ids[offset..offset + length]
                    .iter()
                    .map(|id| {
                        let blob = lookup.query_row([id], |r| r.get::<_, Vec<u8>>(0))?;
                        Ok((id.to_string(), blob))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                stmt.query_map(
                    params![i64::try_from(length)?, i64::try_from(offset)?],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)),
                )?
                .collect::<std::result::Result<Vec<_>, _>>()?
            };
            ensure!(
                records.len() == length,
                "requested range extends beyond the archive"
            );
            let read_us = teachers::elapsed(start);
            let started = Instant::now();
            let output_rows: Vec<_> = pool.install(|| records.into_par_iter().map(|(locator, blob)| {
                let t = Instant::now();
                let locator_hash = source::digest(locator.as_bytes());
                match source::decode(locator, &blob) {
                    Err(e) => json!({"type":"match", "status":"normalize_rejected", "locator_hash":locator_hash,
                        "diagnostic":format!("{e:#}"), "parse_us":teachers::elapsed(t)}),
                    Ok(canonical) => {
                        let parse_us = teachers::elapsed(t);
                        if flags.contains("--normalize-only") {
                            return json!({"type":"match", "status":"normalized", "canonical":canonical, "parse_us":parse_us});
                        }
                        if canonical.rules().is_none() {
                            return json!({"type":"match", "status":"parse_only", "canonical":canonical, "parse_us":parse_us});
                        }
                        match replay::replay(&canonical, !flags.contains("--no-shanten-cache"),
                            !flags.contains("--no-checkpoints"), flags.contains("--measure-restore")) {
                            Ok(mut replay) => {
                                replay.timings.parse_us = parse_us;
                                json!({"type":"match", "status":"replayed", "canonical":canonical, "replay":replay})
                            }
                            Err(e) => json!({"type":"match", "status":"replay_rejected", "canonical":canonical,
                                "diagnostic":format!("{e:#}"), "parse_us":parse_us}),
                        }
                    }
                }
            }).collect());
            let process_us = teachers::elapsed(started);
            if config.format == "jsonl" {
                write_shard(&path, &header, &output_rows, read_us, process_us, &mut seen)?
            } else {
                columnar::write(
                    &path,
                    &header,
                    &output_rows,
                    read_us,
                    process_us,
                    &mut seen,
                    &mut action_registry,
                    &config,
                )?
            }
        };
        eprintln!("range {offset}+{length}: {}", footer["statuses"]);
        if flags.contains("--stop-on-replay-error") {
            for status in ["normalize_rejected", "replay_rejected"] {
                ensure!(
                    footer["statuses"][status].as_u64().unwrap_or(0) == 0,
                    "quality gate stopped at range {offset}+{length} after {status}; shard retained for diagnosis"
                );
            }
        }
        manifests.push(
            json!({"file":path.file_name().unwrap().to_str().unwrap(), "footer":footer,
            "offset":offset, "bytes": if config.format == "jsonl" { path.metadata()?.len() }
            else { footer["files"].as_object().unwrap().values().filter_map(|f| f["bytes"].as_u64()).sum() }}),
        );
    }
    let manifest = json!({"build":build, "shards":manifests, "source_hash_us":source_hash_us});
    let path = output.join("manifest.json");
    if path.exists() {
        let existing: Value = serde_json::from_reader(File::open(&path)?)?;
        ensure!(
            existing["build"] == manifest["build"] && existing["shards"] == manifest["shards"],
            "manifest differs"
        );
    } else {
        atomic_file(&path, |writer| {
            serde_json::to_writer_pretty(writer, &manifest)?;
            Ok(())
        })?;
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn create_private_dir(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.recursive(true).create(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            path.metadata()?.permissions().mode() & 0o077 == 0,
            "output directory must be accessible only by its owner"
        );
    }
    Ok(())
}

fn atomic_file(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let suffix = format!(
        "{}.{}.tmp",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    );
    let temporary: PathBuf = path.with_file_name(suffix);
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut writer = BufWriter::new(opts.open(&temporary)?);
    let result = (|| {
        write(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::hard_link(&temporary, path)?;
        File::open(path.parent().unwrap())?.sync_all()?;
        Ok(())
    })();
    fs::remove_file(&temporary)?;
    result
}

fn row_check(row: &Value, seen: &mut HashSet<String>) -> Result<()> {
    ensure!(row["type"] == "match", "expected match row");
    if let Some(id) = row["canonical"]["id"].as_str() {
        ensure!(seen.insert(id.into()), "duplicate match payload: {id}");
    }
    Ok(())
}

fn write_shard(
    path: &Path,
    header: &Value,
    rows: &[Value],
    read_us: u64,
    process_us: u64,
    seen: &mut HashSet<String>,
) -> Result<Value> {
    let mut sha = Sha256::new();
    let mut statuses = BTreeMap::new();
    let mut footer = Value::Null;
    atomic_file(path, |writer| {
        serde_json::to_writer(&mut *writer, header)?;
        writer.write_all(b"\n")?;
        let t = Instant::now();
        for row in rows {
            row_check(row, seen)?;
            *statuses
                .entry(row["status"].as_str().context("missing status")?)
                .or_insert(0u32) += 1;
            let bytes = serde_json::to_vec(row)?;
            sha.update(&bytes);
            sha.update(b"\n");
            writer.write_all(&bytes)?;
            writer.write_all(b"\n")?;
        }
        footer = json!({"type":"footer","rows":rows.len(),"statuses":statuses,
            "digest":format!("{:x}", sha.finalize()),"read_us":read_us,"process_us":process_us,
            "write_us":teachers::elapsed(t)});
        serde_json::to_writer(&mut *writer, &footer)?;
        writer.write_all(b"\n")?;
        Ok(())
    })?;
    Ok(footer)
}

fn verify_shard(path: &Path, header: &Value, seen: &mut HashSet<String>) -> Result<Value> {
    let reader = BufReader::new(File::open(path)?);
    let mut lines = reader.split(b'\n');
    ensure!(
        serde_json::from_slice::<Value>(&lines.next().context("missing header")??)? == *header,
        "resume settings do not match {path:?}"
    );
    let mut sha = Sha256::new();
    let mut statuses = BTreeMap::new();
    let mut indexed = lines.by_ref().enumerate();
    while let Some((rows, line)) = indexed.next() {
        let line = line?;
        let row: Value = serde_json::from_slice(&line)?;
        if row["type"] == "footer" {
            ensure!(
                row["digest"] == format!("{:x}", sha.finalize())
                    && row["rows"] == rows
                    && row["rows"] == header["requested"]
                    && row["statuses"] == serde_json::to_value(&statuses)?
                    && indexed.next().is_none(),
                "invalid shard footer: {path:?}"
            );
            return Ok(row);
        }
        row_check(&row, seen)?;
        *statuses
            .entry(row["status"].as_str().context("missing status")?.to_owned())
            .or_insert(0u32) += 1;
        sha.update(line);
        sha.update(b"\n");
    }
    bail!("missing footer: {path:?}")
}

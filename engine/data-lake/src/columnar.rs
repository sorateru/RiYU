use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use arrow_array::{RecordBatch, builder::*};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde_json::{Value, json};

use crate::{CONTRACT, hash_file, row_check, source::digest, teachers::elapsed};

const TABLES: &str = include_str!("../../../schemas/riyu-data-lake-pilot-v1.json");

pub struct Config {
    pub format: String,
    pub profile: String,
    pub checkpoint: u64,
    pub batch: usize,
    pub row_group: usize,
    pub zstd: i32,
}

fn values(v: &Value) -> &[Value] {
    v.as_array().map(Vec::as_slice).unwrap_or_default()
}

fn fields(v: &Value) -> Result<Fields> {
    values(v)
        .iter()
        .map(|f| {
            Ok(Arc::new(Field::new(
                f[0].as_str().context("field name")?,
                datatype(&f[1])?,
                true,
            )))
        })
        .collect()
}

fn datatype(v: &Value) -> Result<DataType> {
    Ok(match v.as_str() {
        Some("uint8") => DataType::UInt8,
        Some("uint16") => DataType::UInt16,
        Some("uint32") => DataType::UInt32,
        Some("uint64") => DataType::UInt64,
        Some("int8") => DataType::Int8,
        Some("int32") => DataType::Int32,
        Some("double") => DataType::Float64,
        Some("bool") => DataType::Boolean,
        Some("string") => DataType::Utf8,
        Some("binary") => DataType::Binary,
        Some(s) if s.starts_with('b') => DataType::FixedSizeBinary(s[1..].parse()?),
        _ if v[0] == "list" => DataType::List(Arc::new(Field::new("item", datatype(&v[1])?, true))),
        _ if v[0] == "struct" => DataType::Struct(fields(&v[1])?),
        _ => bail!("unknown Arrow type {v}"),
    })
}

fn unhex(s: &str) -> Result<Vec<u8>> {
    ensure!(s.len() % 2 == 0 && s.is_ascii(), "invalid hex");
    (0..s.len())
        .step_by(2)
        .map(|i| Ok(u8::from_str_radix(&s[i..i + 2], 16)?))
        .collect()
}

fn append(builder: &mut dyn ArrayBuilder, kind: &DataType, v: &Value) -> Result<()> {
    macro_rules! down {
        ($t:ty) => {
            builder
                .as_any_mut()
                .downcast_mut::<$t>()
                .context("builder/type mismatch")?
        };
    }
    macro_rules! scalar {
        ($b:ty, $get:ident) => {{
            let b = down!($b);
            if v.is_null() {
                b.append_null();
            } else {
                b.append_value(v.$get().context("incompatible scalar")?.try_into()?);
            }
        }};
    }
    match kind {
        DataType::UInt8 => scalar!(UInt8Builder, as_u64),
        DataType::UInt16 => scalar!(UInt16Builder, as_u64),
        DataType::UInt32 => scalar!(UInt32Builder, as_u64),
        DataType::UInt64 => down!(UInt64Builder).append_option(v.as_u64()),
        DataType::Int8 => scalar!(Int8Builder, as_i64),
        DataType::Int32 => scalar!(Int32Builder, as_i64),
        DataType::Float64 => down!(Float64Builder).append_option(v.as_f64()),
        DataType::Boolean => down!(BooleanBuilder).append_option(v.as_bool()),
        DataType::Utf8 => down!(StringBuilder).append_option(v.as_str()),
        DataType::Binary => {
            let b = down!(BinaryBuilder);
            if let Some(items) = v.as_array() {
                let bytes = items
                    .iter()
                    .map(|v| Ok(u8::try_from(v.as_u64().context("binary byte")?)?))
                    .collect::<Result<Vec<_>>>()?;
                b.append_value(bytes);
            } else if v.is_null() {
                b.append_null();
            } else {
                b.append_value(serde_json::to_vec(v)?);
            }
        }
        DataType::FixedSizeBinary(_) => {
            let b = down!(FixedSizeBinaryBuilder);
            if let Some(s) = v.as_str() {
                b.append_value(unhex(s)?)?;
            } else if let Some(items) = v.as_array() {
                b.append_value(
                    items
                        .iter()
                        .map(|v| Ok(u8::try_from(v.as_u64().context("fixed binary byte")?)?))
                        .collect::<Result<Vec<_>>>()?,
                )?;
            } else {
                ensure!(v.is_null(), "invalid fixed binary");
                b.append_null();
            }
        }
        DataType::List(field) => {
            let b = down!(ListBuilder<Box<dyn ArrayBuilder>>);
            for item in values(v) {
                append(b.values().as_mut(), field.data_type(), item)?;
            }
            b.append(!v.is_null());
        }
        DataType::Struct(fields) => {
            let b = down!(StructBuilder);
            for (i, field) in fields.iter().enumerate() {
                append(
                    b.field_builders_mut()[i].as_mut(),
                    field.data_type(),
                    &v[field.name()],
                )?;
            }
            b.append(!v.is_null());
        }
        _ => bail!("unsupported builder type {kind:?}"),
    }
    Ok(())
}

enum Writer {
    Parquet(Box<ArrowWriter<File>>),
    Ipc(FileWriter<File>),
}

struct Table {
    schema: SchemaRef,
    columns: Vec<Box<dyn ArrayBuilder>>,
    writer: Option<Writer>,
    rows: usize,
    arrow_us: u64,
    write_us: u64,
}

struct Shard<'a> {
    path: &'a Path,
    config: &'a Config,
    tables: BTreeMap<String, Table>,
    actions: &'a mut HashMap<String, Vec<u8>>,
}

impl Shard<'_> {
    fn add(&mut self, table: &str, sources: &[&Value], overrides: &[(&str, &Value)]) -> Result<()> {
        let target = self.tables.get_mut(table).context("unknown table")?;
        let start = Instant::now();
        for (column, field) in target.columns.iter_mut().zip(target.schema.fields()) {
            let key = field.name().as_str();
            let v = overrides
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, v)| *v)
                .or_else(|| sources.iter().find_map(|v| v.get(key)))
                .unwrap_or(&Value::Null);
            append(column.as_mut(), field.data_type(), v)
                .with_context(|| format!("building {table}.{key}"))?;
        }
        target.rows += 1;
        target.arrow_us += elapsed(start);
        if target.columns[0].len() >= self.config.batch {
            target.flush(self.path, table, self.config)?;
        }
        Ok(())
    }

    fn convert(&mut self, row: &Value) -> Result<()> {
        let m = &row["canonical"];
        let id = &m["id"];
        self.add(
            "restricted/matches",
            &[row, m],
            &[("match_id", id), ("source_locator", &m["locator"])],
        )?;
        let events = values(&m["events"]);
        for e in events {
            let dora = &e["dora"];
            let scores = &e["scores"];
            self.add(
                "restricted/events",
                &[e],
                &[
                    ("match_id", id),
                    ("dora", if dora.is_array() { dora } else { &Value::Null }),
                    (
                        "initial_dora",
                        if dora.is_object() { dora } else { &Value::Null },
                    ),
                    (
                        "score_change",
                        if scores.is_object() {
                            scores
                        } else {
                            &Value::Null
                        },
                    ),
                    (
                        "score_vector",
                        if scores.is_array() {
                            scores
                        } else {
                            &Value::Null
                        },
                    ),
                    ("rating", &e["ratings"]),
                ],
            )?;
        }
        if row["status"] != "replayed" {
            return Ok(());
        }
        let bucket = u64::from_str_radix(&id.as_str().context("match ID")?[..16], 16)? % 100;
        let split = json!(if bucket < 80 {
            "train"
        } else if bucket < 90 {
            "validation"
        } else {
            "final_test"
        });
        self.add(
            "governance/split",
            &[],
            &[("match_id", id), ("split", &split)],
        )?;
        self.add("targets/match", &[m], &[("match_id", id)])?;
        for r in values(&row["replay"]["rounds"]) {
            let round = &r["round_index"];
            let prefix = [("match_id", id), ("round", round)];
            self.add("targets/round", &[r], &prefix)?;
            for ordinal in values(&r["wins"]) {
                let e = events
                    .get(ordinal.as_u64().context("win ordinal")? as usize)
                    .context("missing win")?;
                self.add(
                    "targets/wins",
                    &[e],
                    &[prefix[0], prefix[1], ("score_change", &e["scores"])],
                )?;
            }
            if self.config.profile != "B" {
                for (seat, stream) in values(&r["safe_events"]).iter().enumerate() {
                    for e in values(stream) {
                        self.add(
                            "inputs/events",
                            &[],
                            &[
                                prefix[0],
                                prefix[1],
                                ("seat", &json!(seat)),
                                ("seq", &e["sequence"]),
                                ("event", e),
                            ],
                        )?;
                    }
                }
            }
            if matches!(self.config.profile.as_str(), "C" | "D") {
                for (seat, stream) in values(&r["checkpoints16"]).iter().enumerate() {
                    for c in values(stream) {
                        if c["sequence"].as_u64().context("checkpoint seq")?
                            % self.config.checkpoint
                            == 0
                        {
                            self.add(
                                "inputs/checkpoints",
                                &[],
                                &[
                                    prefix[0],
                                    prefix[1],
                                    ("seat", &json!(seat)),
                                    ("seq", &c["sequence"]),
                                    ("state", &c["state"]),
                                ],
                            )?;
                        }
                    }
                }
            }
            for d in values(&r["decisions"]) {
                self.decision(d, id, round, &m["raw_rule"], events)?;
            }
        }
        Ok(())
    }

    fn decision(
        &mut self,
        d: &Value,
        id: &Value,
        round: &Value,
        rule: &Value,
        events: &[Value],
    ) -> Result<()> {
        let key = ("decision_id", &d["id"]);
        let axis = ("action_set_hash", &d["action_set_hash"]);
        let actions = values(&d["actions"]);
        let mut seen = HashSet::new();
        for a in actions {
            let key = a["id"].as_str().context("action ID")?;
            let semantic: Vec<u8> = serde_json::from_value(a["semantic"].clone())?;
            ensure!(
                seen.insert(key)
                    && self
                        .actions
                        .entry(key.to_owned())
                        .or_insert_with(|| semantic.clone())
                        == &semantic,
                "duplicate/colliding action ID"
            );
        }
        ensure!(
            axis.1 == &digest(&serde_json::to_vec(&d["actions"])?),
            "action axis hash mismatch"
        );
        for label in [&d["attack"]["candidates"], &d["defense"]] {
            let mut last = None;
            for c in values(label) {
                let a = c["action_index"].as_u64().context("teacher action index")? as usize;
                ensure!(
                    a < actions.len() && last.is_none_or(|p| p < a),
                    "invalid teacher action axis"
                );
                last = Some(a);
            }
        }
        let choice = d["choice"].as_u64();
        ensure!(
            choice.is_none_or(|i| (i as usize) < actions.len() && d["choice_status"] != "censored"),
            "invalid policy choice"
        );
        self.add(
            "inputs/decisions",
            &[d],
            &[key, ("match_id", id), ("round", round), ("rule_code", rule)],
        )?;
        if matches!(self.config.profile.as_str(), "B" | "D") {
            self.add(
                "inputs/snapshots",
                &[],
                &[key, ("snapshot", &d["observation"])],
            )?;
        }
        self.add(
            "restricted/execution",
            &[],
            &[key, ("physical_ids", &d["execution"])],
        )?;
        self.add(
            "targets/policy",
            &[d],
            &[
                key,
                axis,
                (
                    "trainable",
                    &json!(choice.is_some() && d["legal_status"] == "complete"),
                ),
            ],
        )?;
        self.add("targets/attack", &[&d["attack"]], &[key, axis])?;
        self.add(
            "targets/defense",
            &[],
            &[key, axis, ("candidates", &d["defense"])],
        )?;
        let next = values(&d["next_opponent_action"]).iter().map(|o| {
            if let Some(ordinal) = o.as_u64() {
                let e = events.get(ordinal as usize).context("missing future event")?;
                let t = &e["tile"];
                let code = t["kind"].as_u64().map(|kind| 2*kind + u64::from(t["red"] == true));
                Ok(json!({"source_ordinal": ordinal, "type": e["type"], "tile_code": code, "meld_kind": e["meld"]["kind"]}))
            } else { Ok(Value::Null) }
        }).collect::<Result<Vec<_>>>()?;
        self.add(
            "targets/belief_lite",
            &[d],
            &[key, ("next_opponent_action", &json!(next))],
        )?;
        let hands = values(&d["opponents"]).iter().map(|h| {
            let mut counts = [0_u8; 34];
            let mut red = 0;
            for tile in values(&h["concealed"]) {
                let kind = tile["kind"].as_u64().context("hidden tile kind")? as usize;
                let slot = counts.get_mut(kind).context("hidden tile out of range")?;
                *slot += 1;
                ensure!(*slot <= 4, "too many hidden tiles");
                if tile["red"] == true { red |= 1 << (kind/9); }
            }
            let t = &h["drawn"];
            let drawn = t["kind"].as_u64().map(|kind| 2*kind + u64::from(t["red"] == true));
            ensure!(counts.iter().map(|&n| usize::from(n)).sum::<usize>() + usize::from(drawn.is_some()) ==
                h["hand_size"].as_u64().context("hidden hand size")? as usize, "hidden hand size mismatch");
            Ok(json!({"seat":h["seat"], "concealed_counts": counts.to_vec(), "red_mask":red, "drawn":drawn}))
        }).collect::<Result<Vec<_>>>()?;
        self.add("targets/belief_full", &[], &[key, ("hands", &json!(hands))])
    }
}

impl Table {
    fn flush(&mut self, root: &Path, name: &str, config: &Config) -> Result<()> {
        if self.columns[0].is_empty() {
            return Ok(());
        }
        let start = Instant::now();
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            self.columns.iter_mut().map(|c| c.finish()).collect(),
        )?;
        self.arrow_us += elapsed(start);
        let start = Instant::now();
        if self.writer.is_none() {
            let path = root.join(name).with_extension(&config.format);
            fs::create_dir_all(path.parent().unwrap())?;
            let file = File::create(path)?;
            self.writer = Some(if config.format == "parquet" {
                let props = WriterProperties::builder()
                    .set_max_row_group_size(config.row_group)
                    .set_compression(Compression::ZSTD(ZstdLevel::try_new(config.zstd)?))
                    .build();
                Writer::Parquet(Box::new(ArrowWriter::try_new(
                    file,
                    self.schema.clone(),
                    Some(props),
                )?))
            } else {
                Writer::Ipc(FileWriter::try_new(file, &self.schema)?)
            });
        }
        match self.writer.as_mut().unwrap() {
            Writer::Parquet(w) => w.write(&batch)?,
            Writer::Ipc(w) => w.write(&batch)?,
        }
        self.write_us += elapsed(start);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write(
    dir: &Path,
    header: &Value,
    rows: &[Value],
    read_us: u64,
    process_us: u64,
    seen: &mut HashSet<String>,
    actions: &mut HashMap<String, Vec<u8>>,
    config: &Config,
) -> Result<Value> {
    let t = Instant::now();
    let tmp = dir.with_extension(format!("{}.partial", std::process::id()));
    fs::create_dir(&tmp)?;
    let spec: BTreeMap<String, Value> = serde_json::from_str(TABLES)?;
    let mut tables = BTreeMap::new();
    for (name, info) in spec {
        let schema = Arc::new(Schema::new_with_metadata(
            fields(&info["fields"])?,
            HashMap::from([
                ("riyu.contract".into(), CONTRACT.into()),
                ("riyu.role".into(), name.split('/').next().unwrap().into()),
                (
                    "riyu.origin".into(),
                    info["origin"].as_str().context("table origin")?.into(),
                ),
            ]),
        ));
        let columns = schema
            .fields()
            .iter()
            .map(|f| make_builder(f.data_type(), 0))
            .collect();
        tables.insert(
            name,
            Table {
                schema,
                columns,
                writer: None,
                rows: 0,
                arrow_us: 0,
                write_us: 0,
            },
        );
    }
    let mut shard = Shard {
        path: &tmp,
        config,
        tables,
        actions,
    };
    let mut statuses = BTreeMap::new();
    let mut ids = Vec::new();
    let mut timing = BTreeMap::new();
    let mut restore_ns = Vec::new();
    for row in rows {
        row_check(row, seen)?;
        if let Some(id) = row["canonical"]["id"].as_str() {
            ids.push(id);
        }
        *statuses
            .entry(row["status"].as_str().context("missing status")?)
            .or_insert(0u32) += 1;
        if let Some(values) = row["replay"]["timings"].as_object() {
            for (k, v) in values {
                *timing.entry(k).or_insert(0u64) += v.as_u64().unwrap_or(0);
            }
        }
        restore_ns.extend(values(&row["replay"]["restore"]));
        shard.convert(row)?;
    }
    let mut files = BTreeMap::new();
    for (name, table) in &mut shard.tables {
        table.flush(&tmp, name, config)?;
        let start = Instant::now();
        match table.writer.take() {
            Some(Writer::Parquet(w)) => {
                w.into_inner()?.sync_all()?;
            }
            Some(Writer::Ipc(mut w)) => {
                w.finish()?;
                w.get_ref().sync_all()?;
            }
            None => continue,
        }
        table.write_us += elapsed(start);
        let p = tmp.join(name).with_extension(&config.format);
        files.insert(
            name.as_str(),
            json!({
                "rows":table.rows, "bytes":p.metadata()?.len(), "sha256":hash_file(&p)?,
                "arrow_us":table.arrow_us, "encode_write_us":table.write_us
            }),
        );
    }
    let footer = json!({"header":header,"rows":rows.len(), "statuses":statuses,"ids":ids,
        "files":files,"timings":timing,"restore_ns":restore_ns,"read_us":read_us,"process_us":process_us,
        "write_us":elapsed(t)});
    crate::atomic_file(&tmp.join("commit.json"), |w| {
        serde_json::to_writer(w, &footer)?;
        Ok(())
    })?;
    fs::rename(&tmp, dir)?;
    File::open(dir.parent().unwrap())?.sync_all()?;
    Ok(footer)
}

pub fn verify(
    dir: &Path,
    header: &Value,
    seen: &mut HashSet<String>,
    format: &str,
) -> Result<Value> {
    let footer: Value = serde_json::from_reader(File::open(dir.join("commit.json"))?)?;
    ensure!(
        footer["header"] == *header && footer["rows"] == header["requested"],
        "columnar resume configuration changed"
    );
    let spec: BTreeMap<String, Value> = serde_json::from_str(TABLES)?;
    let files = footer["files"]
        .as_object()
        .context("missing committed files")?;
    ensure!(
        files
            .get("restricted/matches")
            .is_some_and(|f| f["rows"] == footer["rows"]),
        "missing match rows"
    );
    for (name, f) in files {
        ensure!(spec.contains_key(name), "unexpected table");
        let path = dir.join(name).with_extension(format);
        ensure!(
            f["bytes"] == path.metadata()?.len() && f["sha256"] == hash_file(&path)?,
            "invalid table checksum: {name}"
        );
    }
    for id in values(&footer["ids"]) {
        ensure!(
            seen.insert(id.as_str().context("match ID")?.to_owned()),
            "duplicate match"
        );
    }
    Ok(footer)
}

from storage.s3_storage import S3Storage
from preprocess.cleaners import normalize_text, extract_raw_units
from preprocess.db_writer import PostgresWriter
from preprocess.llm_placeholders import SpecKeyEnricher


def parse_s3_uri(uri: str) -> tuple[str | None, str]:
    path = uri.replace("s3://", "", 1)
    if "/" not in path:
        return None, path
    bucket, key = path.split("/", 1)
    return bucket, key


def process_single_key(
    s3: S3Storage,
    writer: PostgresWriter,
    key: str,
):
    """
    Xử lý 1 file JSON duy nhất
    → build spec_map + unit_map
    → insert DB
    → giải phóng RAM
    """
    print(f"Processing {key}")

    spec_map = {}
    unit_map = {}

    for doc in s3.stream_json(key):
        product = doc.get("content", {}).get("product", {})
        if not isinstance(product, dict):
            continue

        for raw_key, raw_value in product.items():
            # ---- spec key ----
            norm_key = normalize_text(raw_key)
            spec_map[norm_key] = (raw_key, norm_key)

            # ---- unit ----
            if isinstance(raw_value, str):
                for u in extract_raw_units(raw_value):
                    unit_map[u] = (u, normalize_text(u))

    if spec_map:
        writer.insert_spec_keys(list(spec_map.values()))

    if unit_map:
        writer.insert_raw_units(list(unit_map.values()))

    print(
        f"Inserted {len(spec_map)} spec keys, "
        f"{len(unit_map)} raw units"
    )


def run(s3_uri: str):
    bucket, key_or_prefix = parse_s3_uri(s3_uri)

    s3 = S3Storage()
    if bucket:
        s3.bucket = bucket

    writer = PostgresWriter()

    # ---- CASE 1: single file ----
    if key_or_prefix.endswith(".json"):
        process_single_key(s3, writer, key_or_prefix)
        return

    # ---- CASE 2: prefix ----
    print(f"Listing files under s3://{s3.bucket}/{key_or_prefix}")
    keys = s3.list_keys(key_or_prefix)

    json_keys = [k for k in keys if k.endswith(".json")]
    print(f"Found {len(json_keys)} json files")

    for key in json_keys:
        process_single_key(s3, writer, key)
    
def enrich_spec_keys(batch_size: int = 20):
    writer = PostgresWriter()
    enricher = SpecKeyEnricher(batch_size=batch_size)

    while True:
        # Lấy batch chưa enrich
        rows = writer.fetch_spec_keys_missing_enrichment(batch_size)

        if not rows:
            print("No more spec keys to enrich")
            break

        print(f"Enriching {len(rows)} spec keys")

        enriched = enricher.enrich_batch(rows)

        writer.update_spec_key_enrichment(enriched)


if __name__ == "__main__":
    # run("s3://20251-data-science/output/")
    enrich_spec_keys(batch_size=20)

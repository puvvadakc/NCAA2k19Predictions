from google.cloud import bigquery

client = bigquery.Client()

ncaa_dataset_ref = client.dataset('ncaa_basketball', project='bigquery-public-data')

ncaa_dataset = client.get_dataset(hn_dataset_ref)

print([x.table_id for x in client.list_tables(ncaa_dataset)])
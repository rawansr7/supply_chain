# supply_chain

Django app for store-level demand forecasting: upload a sales history, place your stores
on a map, and get next-month order quantities from Chronos-2 plus a newsvendor rule.

## Requirements

- Python 3.11, then `pip install -r requirements.txt`
- A Google Maps **JavaScript API** key, restricted to that API and to the referrer
  `http://localhost:8000/*`
- The first forecast downloads the Chronos-2 weights (~450 MB) from Hugging Face. CPU is
  enough.

## Running

```bash
export GOOGLE_MAPS_API_KEY=<your key>   # read from the environment; never commit it
./run_server.sh                         # migrate, collectstatic, serve on :8000
```

Sign up at <http://localhost:8000/>, register a company, then upload a CSV with the
columns `item_id, store_id, sold, date`. `./reset_db.sh` starts over from an empty
database.

The forecasting code lives in [`analysis/`](analysis/README.md).

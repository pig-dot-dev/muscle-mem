import duckdb
import time
import json

con = duckdb.connect()

con.execute("CREATE TABLE items (timestamp TIMESTAMP, data JSON)")

# Load db
in_mem = []
t0 = time.time()
count = 10_000
print("Loading db")
for i in range(count):
    data = json.dumps({
        "hi": "erik",
        "count": i
    })
    in_mem.append(data)
    con.execute("INSERT INTO items (timestamp, data) VALUES (CURRENT_TIMESTAMP, ?)", [data])
print("Time:", time.time() - t0)

# Query db
results = con.execute("SELECT * FROM items LIMIT 5").fetchall()
print("First 5 items:", results)

def udf(current_data, candidate_data):
    current = json.loads(current_data)
    candidate = json.loads(candidate_data)
    return current['count'] == candidate['count']

# Register the UDF with DuckDB
con.create_function("match_count", udf, [duckdb.typing.VARCHAR, duckdb.typing.VARCHAR], duckdb.typing.BOOLEAN)

# Create a new data object with count = 100
current_data = json.dumps({
    "hi": "erik",
    "count": 100
})

# Query using the UDF to find the matching record
t0 = time.time()
results = con.execute("""
    SELECT * FROM items 
    WHERE match_count(?, data) = true
""", [current_data]).fetchall()

print("Found matching record with count = 100:")
print(results)
print("Time:", time.time() - t0)

# Query in memory
t0 = time.time()
res = None
for i in range(len(in_mem)):
    if udf(current_data, in_mem[i]):
        res = in_mem[i]
        
print("Found matching record with count = 100:")
print(res)
print("Time:", time.time() - t0)


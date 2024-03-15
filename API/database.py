import sqlite3
from io import BytesIO
from pathlib import Path
import numpy as np

file_path = Path("data.db")

# create scheme if loaded with empty database
connection = sqlite3.connect(str(file_path))
cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS "embeddings" (
	"name"	TEXT NOT NULL UNIQUE,
	"embedding"	BLOB NOT NULL,
	"id"	INTEGER NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);
""")
connection.commit()
connection.close()

def insert_new_object(name, vector):
    connection = sqlite3.connect(str(file_path))
    cursor = connection.cursor()
    byte_stream = BytesIO()  # create a new byte stream

    # save vector as .npy into the in-memory byte stream
    np.save(file=byte_stream, arr=vector)

    # transform vector to bytes and save to sqlite database as blob
    try:
        cursor.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (?, ?)",
            [name, byte_stream.getvalue()],
        )
        connection.commit()  # save changes
    except sqlite3.IntegrityError:
        connection.close()
        return False
    else:
        connection.close()
        return True
    

    
def find_stamp_group(name):
    connection = sqlite3.connect(str(file_path))
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM embeddings WHERE name LIKE ?;", [name + "%"])
    stamp_names = cursor.fetchall()
    connection.close()
    return stamp_names


def cosine_similarity(x, y):
    # Ensure length of x and y are the same
    if len(x) != len(y):
        return None

    # Compute the dot product between x and y
    dot_product = np.dot(x, y)

    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sqrt(np.sum(x**2))
    magnitude_y = np.sqrt(np.sum(y**2))

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)

    return cosine_similarity


def find_max_cosine_similarity(vector):
    # fetch everything from sqlite
    connection = sqlite3.connect(str(file_path))
    cursor = connection.cursor()
    cursor.execute("SELECT id, name, embedding FROM embeddings")
    raw_data = cursor.fetchall()
    connection.close()

    # the following code has low performace and high memory consumption,
    # but it is required, since sqlite can not perform cosine similarity

    # create byte stream from embeddings and load as numpy array for each entity
    data = [
        {"id": id, "name": name, "embedding": np.load(file=BytesIO(embedding))}
        for (id, name, embedding) in raw_data
    ]

    # find entity by custom function (e.g. cosine_similarity with given vector)
    best_match = max(data, key=lambda x: cosine_similarity(x["embedding"], vector))
    accuracy = cosine_similarity(best_match["embedding"], vector)
    
    if accuracy >= 0.85:
        return {
            "stamp_name": best_match["name"],
            "accuracy": float(accuracy),
        }

    return None

    # best_match is python dict:
    # {"id": *int*, "name": *string*, "embeddings": *np.array*}

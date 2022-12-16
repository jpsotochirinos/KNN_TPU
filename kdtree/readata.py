from time import process_time
import jax.numpy as jnp
import csv
data = []
with open('/home/lordcocoro2004/maestria/Recuperaciondelainfo/ml-latest-small/movie_rating.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        row
        break
    for row in reader:
        data.append([0 if value=='' else int(value) for value in row[1:]])
        #print(row[1:])

print(jnp.array(data))
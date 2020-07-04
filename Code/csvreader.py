import csv
wr=""
r = csv.reader(open('ndvi.csv')) # Here your csv file
lines = list(r)
for row in lines:
    row[0]=row[0][0:4]+"-"+row[0][4:6]+"-"+row[0][6:]

writer = csv.writer(open('ndvi.csv', 'w'))
writer.writerows(lines)


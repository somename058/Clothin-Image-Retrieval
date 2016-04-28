import json
f=open('final3.txt','a')

	
results={}
f2=open('results3.txt','r')
for i in range(0,12):
	s=f2.readline()
	d=json.loads(s)
	for j in range(0,100):
		for k in d[j].keys():
			if k not in results.keys():
				results[k]=[j]
			else:
				results[k].append(j)

for k in results.keys():
	if len(results[k])==12:
		f.write(k+':'+str(results[k])+'\n')
f.close()
f2.close()				
 

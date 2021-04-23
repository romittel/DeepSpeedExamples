import json
import os
data_dir = "/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/synthetic_data"
num_of_files = 5
num_of_lines = 10000


for i in range(num_of_files):
    path = os.path.join(data_dir, "file_" + str(i) + ".txt")
    f = open(path, "w", encoding='utf8')
    a = {}
    a["task_id"] = str(i)
    a["docs"] = [{"title": "the asch conformity experiments - verywell mind", "snippet": "", "url": "www.verywellmind.com/the-asch-conformity-experiments-2794996", "qbclicks": "0", "sqbclicks": "0", "noclicks": "11", "satclicks": "0", "utility": "-1.37359488179406", "hrslabel": "4.66666666666667", "occurance": "11", "gclickrank": "-1", "gclickcount": "0", "language": "en", "region": "au", "query": "conformity experiment", "marketaffinity": "0.320099999999997", "instapop": "0.0905643273341706"} for _ in range(4)]
    for j in range(num_of_lines):
        f.write(json.dumps(a, ensure_ascii=False) + "\n")
    f.close()

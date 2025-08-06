from pysolotools.consumers import Solo
solo = Solo(data_path="E:\softwares_document\VS_Code_Projects\python_Project\FreeAskWorldSimulator\data\solo")

for frame in solo.frames():
    print(frame)
    break
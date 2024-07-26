Esse modelo é capaz de prever as tarifas dos taxis de Nova Iorque (Nos anos de 2010) com base em parâmetros como a distância percorrida em kilômetros, o horário, o dia da semana, o horário em AM/PM, e a data. 

Possui 8 camadas, essas são: 
    (0): Linear(in_features=23, out_features=300, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.4, inplace=False)
    (4): Linear(in_features=300, out_features=200, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.4, inplace=False)
    (8): Linear(in_features=200, out_features=1, bias=True)

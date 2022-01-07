from Application1_cutting_planes import main_app1
from Application2_cutting_planes import main_app2

#Call application 1

for i in range(1,11):
    name = "PSD_random"+str(i)
    main_app1(name)
    name = "notPSD_random"+str(i)
    main_app1(name)

#Call application 2
list_graphs = ["jean","myciel4","myciel5","myciel6","myciel7","queen5_5","queen6_6","queen7_7","queen8_8","queen8_12","queen9_9"]

for graph in list_graphs:
    nameDimacs = graph+".col"
    name = graph+"_det1"
    main_app2(nameDimacs,name)
    name = graph+"_random1"
    main_app2(nameDimacs,name)

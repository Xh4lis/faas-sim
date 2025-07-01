from sim.faas import FunctionResourceCharacterization

# python-pi was made with req=10000, except xeongpu - this one was req=2000, due to the inexplicable slow response
ai_resources_per_node_image = {
    ("rockpi", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.26066751860455545, 0.0, 0, 867.0644652888737, 0.02850091195999672
    ),
    ("rockpi", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.029841064731323804,
        4770376.648598975,
        0,
        56.153155728656486,
        0.28409994931238014,
    ),
    ("rockpi", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.470676860647695, 0.0, 0, 6595019.673450895, 0.15783981610514522
    ),
    (
        "rockpi",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.16995227926443168, 0.0, 0, 95857.20685534734, 0.027038749077793266
    ),
    (
        "rockpi",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.3433166487566248, 0.0, 0, 21192217.072344374, 0.03577910586933355
    ),
    ("rockpi", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.1621359802529322, 0.0, 0, 3210842.724971482, 0.04184853167040104
    ),
    ("tx2", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.3707047087672767, 0.0, 0.0, 1810.5183292303982, 0.02515170415577445
    ),
    ("tx2", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.12237677348449455,
        30368386.89545316,
        0.0,
        273.162570864271,
        0.0996017719111659,
    ),
    ("tx2", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.6437372515048979, 0.0, 0.0, 14922976.382232904, 0.0892017810225266
    ),
    ("tx2", "faas-workloads/resnet-inference-gpu"): FunctionResourceCharacterization(
        0.5056727639364579,
        0.0,
        0.040202020202020204,
        26959145.991050187,
        0.42826641528347176,
    ),
    ("tx2", "faas-workloads/speech-inference-gpu"): FunctionResourceCharacterization(
        0.24903698450299938,
        0.0,
        0.20655083333333335,
        153013.5103681544,
        0.19720434172903806,
    ),
    ("tx2", "faas-workloads/speech-inference-tflite"): FunctionResourceCharacterization(
        0.2705111164968124, 0.0, 0.0, 107222.56215796615, 0.020603045265239994
    ),
    (
        "tx2",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.4896754723162535, 0.0, 0.0, 32551253.14794654, 0.029972260876479768
    ),
    ("tx2", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.25188175990905504, 0.0, 0.0, 3878847.7339153336, 0.03050730938205095
    ),
    ("tx2", "faas-workloads/resnet-training-gpu"): FunctionResourceCharacterization(
        0.1788531851032265,
        205334.00746910932,
        0.7416627493881441,
        1140886.153462762,
        0.7747928186897689,
    ),
    ("tx2", "faas-workloads/tf-gpu"): FunctionResourceCharacterization(
        0.0592452081327536,
        0.0,
        0.5743916666666666,
        1928.7868725481815,
        0.2680824975730648,
    ),
    ("nano", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.4176991404969024, 0.0, 0.0, 1178.4081169827807, 0.029120248518202826
    ),
    ("nano", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.056646811885180295,
        6683200.396891743,
        0.0,
        62.77604036087152,
        0.28360788708505186,
    ),
    ("nano", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.6173617438212513, 0.0, 0.0, 11929148.603204524, 0.16747364790626862
    ),
    ("nano", "faas-workloads/resnet-inference-gpu"): FunctionResourceCharacterization(
        0.3093157264863544,
        303260.485031949,
        0.1754387755102041,
        14102879.228728207,
        0.7423479015527812,
    ),
    ("nano", "faas-workloads/speech-inference-gpu"): FunctionResourceCharacterization(
        0.18332872406064035,
        286.01805267962396,
        0.3609263333333333,
        131431.49378338552,
        0.3460782388908323,
    ),
    (
        "nano",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.25193417112173455, 0.0, 0.0, 93917.27007907239, 0.05844299798485439
    ),
    (
        "nano",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.4648267977623851, 0.0, 0.0, 22159311.443426955, 0.06825314958884132
    ),
    ("nano", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.23927812156043296, 0.0, 0.0, 3129712.5539431805, 0.07892593510475636
    ),
    ("nano", "faas-workloads/resnet-training-gpu"): FunctionResourceCharacterization(
        0.09477461470306936,
        16690880.000560522,
        0.614075023966652,
        445353.4862114798,
        0.8325312889088585,
    ),
    ("nano", "faas-workloads/tf-gpu"): FunctionResourceCharacterization(
        0.06779100962131734, 0.0, 0.076765, 17660.26742032455, 0.45875627577440226
    ),
    ("nx", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.2736460766287605, 0.0, 0.0, 1007.5886562769618, 0.014838573803879939
    ),
    ("nx", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.030209142703148596,
        9512751.066545032,
        0.0,
        88.20910117155238,
        0.1256662577911986,
    ),
    ("nx", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.32085762427108155, 0.0, 0.0, 16688952.186395567, 0.08214067642582418
    ),
    ("nx", "faas-workloads/resnet-inference-gpu"): FunctionResourceCharacterization(
        0.32610813751087114,
        0.0,
        0.036719999999999996,
        26647216.18205711,
        0.40646431091783125,
    ),
    ("nx", "faas-workloads/speech-inference-gpu"): FunctionResourceCharacterization(
        0.1528258733857839,
        0.0,
        0.20816166666666663,
        280619.2978975475,
        0.12338208555916783,
    ),
    ("nx", "faas-workloads/speech-inference-tflite"): FunctionResourceCharacterization(
        0.16536599515876219, 0.0, 0.0, 137404.4579713001, 0.012394685441082152
    ),
    (
        "nx",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.2725498346539076, 0.0, 0.0, 26977091.707812138, 0.015809704470836124
    ),
    ("nx", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.15608878455355574, 0.0, 0.0, 4179408.7300580596, 0.019304396503614804
    ),
    ("nx", "faas-workloads/resnet-training-gpu"): FunctionResourceCharacterization(
        0.13539556486740242,
        33036.85715551058,
        0.6788861615628918,
        5720992.820388119,
        0.7181088281389225,
    ),
    ("nx", "faas-workloads/tf-gpu"): FunctionResourceCharacterization(
        0.021658557716071877,
        0.0,
        0.5042683333333333,
        3066.2566823401226,
        0.28548891539449794,
    ),
    ("rpi4", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.2589641851156356,
        98.77116879952688,
        0,
        37.282525984785686,
        0.08304058474398496,
    ),
    ("rpi4", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.05734476965774494,
        2212988.4169104667,
        0,
        51.35463515857114,
        0.5179107225468366,
    ),
    ("rpi4", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.6178520697554751,
        3493.9240103304014,
        0,
        2788183.4486825834,
        0.31247270825065837,
    ),
    (
        "rpi4",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.25404465742300353,
        181.77953125389715,
        0,
        55027.72837537202,
        0.08083058179679631,
    ),
    (
        "rpi4",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.3859871775412303, 729.0928823356834, 0, 8323463.029890781, 0.09021695908542475
    ),
    ("rpi4", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.25124224587596994, 55.58123936730799, 0, 1346582.9711275853, 0.087515099187977
    ),
    ("nuc", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.16327930662578122,
        5369.999178396372,
        0,
        4335.164300377482,
        0.0072464424825868495,
    ),
    ("nuc", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.08518180777178165,
        443474371.93204427,
        0,
        1058.6437760590277,
        0.033604757826587416,
    ),
    ("nuc", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.439330704824215, 35006.21431055329, 0, 55698244.64783099, 0.04623383643545283
    ),
    ("nuc", "faas-workloads/speech-inference-tflite"): FunctionResourceCharacterization(
        0.12473702325767871, 0.0, 0, 377237.4841574662, 0.014440833755648879
    ),
    (
        "nuc",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.21125373689115415,
        15348.224036911932,
        0,
        38885237.05583308,
        0.028988394596170506,
    ),
    ("nuc", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.10658335529885898,
        1695.0086579905637,
        0,
        10642157.205724549,
        0.017354641403357837,
    ),
    ("nuc", "faas-workloads/resnet-training-cpu"): FunctionResourceCharacterization(
        0.8778859671650178, 229621.9812613565, 0, 1416910.5880434313, 0.5770261788143544
    ),
    ("xeongpu", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.25104879045872724, 0.0, 0.0, 27.438777768693512, 0.004305198283637087
    ),
    ("xeoncpu", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.25104879045872724, 0.0, 0.0, 27.438777768693512, 0.004305198283637087
    ),
    ("xeongpu", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.17225484698783156,
        105923185.84862246,
        0.0,
        1033.0406424383216,
        0.03229357762155556,
    ),
    ("xeoncpu", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.17225484698783156,
        105923185.84862246,
        0.0,
        1033.0406424383216,
        0.03229357762155556,
    ),
    (
        "xeongpu",
        "faas-workloads/resnet-inference-cpu",
    ): FunctionResourceCharacterization(
        0.7157778904909182, 0.0, 0.0, 63570485.920656934, 0.03538363445374544
    ),
    (
        "xeoncpu",
        "faas-workloads/resnet-inference-cpu",
    ): FunctionResourceCharacterization(
        0.7157778904909182, 0.0, 0.0, 63570485.920656934, 0.03538363445374544
    ),
    (
        "xeongpu",
        "faas-workloads/resnet-inference-gpu",
    ): FunctionResourceCharacterization(
        0.37430461089807654, 0.0, 0.005, 57658305.02172596, 0.1259818535384398
    ),
    (
        "xeongpu",
        "faas-workloads/speech-inference-gpu",
    ): FunctionResourceCharacterization(
        0.28122237682203244,
        1847.2475203267718,
        0.08349999999999999,
        513443.86782282026,
        0.044492129917988656,
    ),
    (
        "xeongpu",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.2532600877648794, 0.0, 0.0, 346503.29718918464, 0.0037081284644138713
    ),
    (
        "xeoncpu",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.2532600877648794, 0.0, 0.0, 346503.29718918464, 0.0037081284644138713
    ),
    (
        "xeongpu",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.41735802291597573, 0.0, 0.0, 33404170.874672726, 0.018836604051280674
    ),
    (
        "xeoncpu",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.41735802291597573, 0.0, 0.0, 33404170.874672726, 0.018836604051280674
    ),
    (
        "xeongpu",
        "faas-workloads/resnet-preprocessing",
    ): FunctionResourceCharacterization(
        0.20870127200306265,
        526.0311884296032,
        0.0,
        10226057.56153548,
        0.07645132585138097,
    ),
    (
        "xeoncpu",
        "faas-workloads/resnet-preprocessing",
    ): FunctionResourceCharacterization(
        0.20870127200306265,
        526.0311884296032,
        0.0,
        10226057.56153548,
        0.07645132585138097,
    ),
    ("xeongpu", "faas-workloads/resnet-training-gpu"): FunctionResourceCharacterization(
        0.24106662185734815,
        12032.826369487575,
        0.6002495136081041,
        15750963.289374296,
        0.2848634727401207,
    ),
    ("xeongpu", "faas-workloads/tf-gpu"): FunctionResourceCharacterization(
        0.24361773481853854, 0.0, 0.0565, 9723.279356785817, 0.06644044468790623
    ),
    ("coral", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.35645089819143094,
        735752.0267285869,
        0,
        204.01414233371494,
        0.5667724395036686,
    ),
    ("coral", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.3790727085613829, 22642005.729414783, 0, 5099707.807677874, 0.5559619094971713
    ),
    (
        "coral",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.2501691236630753, 0.0, 0, 50410.417657074395, 0.08785061064186064
    ),
    (
        "coral",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.27407894105734276,
        4642.389351207664,
        0,
        15975138.88173663,
        0.12149107411691996,
    ),
    (
        "coral",
        "faas-workloads/mobilenet-inference-tpu",
    ): FunctionResourceCharacterization(
        0.3544534661066998, 0.0, 0, 19733715.5402467, 0.19948076953552127
    ),
    ("coral", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.24167246170046083,
        2160.2060951038548,
        0,
        2124329.7587075434,
        0.14842207833369112,
    ),
    ("rpi3", "faas-workloads/python-pi"): FunctionResourceCharacterization(
        0.2632196988891948, 0.0, 0, 14.415714537708084, 0.06673025668128908
    ),
    ("rpi3", "faas-workloads/fio"): FunctionResourceCharacterization(
        0.06289025503339646,
        2225277.087690125,
        0,
        50.797641637527974,
        0.6536923071250603,
    ),
    ("rpi3", "faas-workloads/resnet-inference-cpu"): FunctionResourceCharacterization(
        0.5448381974407112,
        2615.734632324097,
        0,
        2062539.5723288062,
        0.36519032690251113,
    ),
    (
        "rpi3",
        "faas-workloads/speech-inference-tflite",
    ): FunctionResourceCharacterization(
        0.25443755203171387,
        64.46054557083016,
        0,
        22390.46618188261,
        0.12244167296212195,
    ),
    (
        "rpi3",
        "faas-workloads/mobilenet-inference-tflite",
    ): FunctionResourceCharacterization(
        0.3035938910644207,
        60.130661576483014,
        0,
        5175658.184667777,
        0.13787648928091437,
    ),
    ("rpi3", "faas-workloads/resnet-preprocessing"): FunctionResourceCharacterization(
        0.23920668104893472,
        45.88347111936155,
        0,
        610213.8222734069,
        0.12884635421827495,
    ),
        # VIDEO ANALYTICS - Similar to resnet-inference (high CPU, moderate I/O)
    ("xeoncpu", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.4393307048, blkio=35006.214, gpu=0, net=55698244.647, ram=0.046233836
    ),
    ("rpi4", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.6178520697, blkio=3493.924, gpu=0, net=2788183.448, ram=0.312472708
    ),
    ("rpi3", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.5448381974, blkio=2615.734, gpu=0, net=2062539.572, ram=0.365190326
    ),
    ("tx2", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.6437372515, blkio=0.0, gpu=0.0, net=14922976.382, ram=0.089201781
    ),
    ("nx", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.3208576242, blkio=0.0, gpu=0.0, net=16688952.186, ram=0.082140676
    ),
    ("nano", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.6173617438, blkio=0.0, gpu=0.0, net=11929148.603, ram=0.167473647
    ),
    ("coral", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.3790727085, blkio=22642005.729, gpu=0, net=5099707.807, ram=0.555961909
    ),
    ("rockpi", "faas-workloads/video-analytics"): FunctionResourceCharacterization(
        cpu=0.4706768606, blkio=0.0, gpu=0, net=6595019.673, ram=0.157839816
    ),
    
    # IOT DATA PROCESSOR - Similar to python-pi (lightweight processing)
    ("xeoncpu", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.2510487904, blkio=0.0, gpu=0.0, net=27.438777768, ram=0.004305198
    ),
    ("rpi4", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.2589641851, blkio=98.771168799, gpu=0, net=37.282525984, ram=0.083040584
    ),
    ("rpi3", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.2632196988, blkio=0.0, gpu=0, net=14.415714537, ram=0.066730256
    ),
    ("tx2", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.3707047087, blkio=0.0, gpu=0.0, net=1810.518329, ram=0.025151704
    ),
    ("nx", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.2736460766, blkio=0.0, gpu=0.0, net=1007.588656, ram=0.014838573
    ),
    ("nano", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.4176991404, blkio=0.0, gpu=0.0, net=1178.408116, ram=0.029120248
    ),
    ("coral", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.3564508981, blkio=735752.026, gpu=0, net=204.014142, ram=0.566772439
    ),
    ("rockpi", "faas-workloads/iot-data-processor"): FunctionResourceCharacterization(
        cpu=0.2606675186, blkio=0.0, gpu=0, net=867.064465, ram=0.028500911
    ),
    
    # ALERT SERVICE - Similar to speech-inference-tflite (fast, lightweight)
    ("xeoncpu", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2532600877, blkio=0.0, gpu=0.0, net=346503.297, ram=0.003708128
    ),
    ("rpi4", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2540446574, blkio=181.779531, gpu=0, net=55027.728375, ram=0.080830581
    ),
    ("rpi3", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2544375520, blkio=64.460545570, gpu=0, net=22390.466181, ram=0.122441672
    ),
    ("tx2", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2705111164, blkio=0.0, gpu=0.0, net=107222.562157, ram=0.020603045
    ),
    ("nx", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.1653659951, blkio=0.0, gpu=0.0, net=137404.457971, ram=0.012394685
    ),
    ("nano", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2519341711, blkio=0.0, gpu=0.0, net=93917.270079, ram=0.058442997
    ),
    ("coral", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.2501691236, blkio=0.0, gpu=0, net=50410.417657, ram=0.087850610
    ),
    ("rockpi", "faas-workloads/alert-service"): FunctionResourceCharacterization(
        cpu=0.1699522792, blkio=0.0, gpu=0, net=95857.206855, ram=0.027038749
    ),
    
    # DATA AGGREGATOR - Similar to resnet-preprocessing (data processing)
    ("xeoncpu", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2087012720, blkio=526.031188429, gpu=0.0, net=10226057.561, ram=0.076451325
    ),
    ("rpi4", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2512422458, blkio=55.581239367, gpu=0, net=1346582.971, ram=0.087515099
    ),
    ("rpi3", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2392066810, blkio=45.883471119, gpu=0, net=610213.822273, ram=0.128846354
    ),
    ("tx2", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2518817599, blkio=0.0, gpu=0.0, net=3878847.733, ram=0.030507309
    ),
    ("nx", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.1560887845, blkio=0.0, gpu=0.0, net=4179408.730, ram=0.019304396
    ),
    ("nano", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2392781215, blkio=0.0, gpu=0.0, net=3129712.553, ram=0.078925935
    ),
    ("coral", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.2416724617, blkio=2160.206095, gpu=0, net=2124329.758, ram=0.148422078
    ),
    ("rockpi", "faas-workloads/data-aggregator"): FunctionResourceCharacterization(
        cpu=0.1621359802, blkio=0.0, gpu=0, net=3210842.724, ram=0.041848531
    ),
}

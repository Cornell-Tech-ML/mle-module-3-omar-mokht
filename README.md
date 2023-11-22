[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)
# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py


**GRAPH**

<img width="719" alt="Screen Shot 2023-11-21 at 11 16 52 PM" src="https://github.com/Cornell-Tech-ML/mle-module-3-omar-mokht/assets/111816253/4f62ffc5-da8e-4b4e-9202-6b40fdead0be">

<img width="474" alt="Screen Shot 2023-11-21 at 11 17 22 PM" src="https://github.com/Cornell-Tech-ML/mle-module-3-omar-mokht/assets/111816253/b7429e51-0ab6-458c-b316-0667b832b220">

*************************************************************************************************************************************************************

[Parallel_Check](log_files/parallel_check_logs.txt)

**CPU SIMPLE 100**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  0  loss  6.033637485361198 correct 46 Time for this one epoch:  14.555243015289307
Epoch  10  loss  2.2422363504448293 correct 49 Time for this one epoch:  0.10104680061340332
Epoch  20  loss  1.6424511273234785 correct 48 Time for this one epoch:  0.09991288185119629
Epoch  30  loss  1.1713088600441475 correct 49 Time for this one epoch:  0.09585285186767578
Epoch  40  loss  0.35457326141524276 correct 49 Time for this one epoch:  0.09650278091430664
Epoch  50  loss  0.2858437693628765 correct 49 Time for this one epoch:  0.09775185585021973
Epoch  60  loss  0.5277299600234142 correct 50 Time for this one epoch:  0.10622882843017578
Epoch  70  loss  0.7154942029121256 correct 49 Time for this one epoch:  0.09718799591064453
Epoch  80  loss  0.3709413435061348 correct 50 Time for this one epoch:  0.09573888778686523
Epoch  90  loss  1.0423649469559495 correct 50 Time for this one epoch:  0.09645605087280273
Epoch  100  loss  0.09797272555546659 correct 50 Time for this one epoch:  0.09554409980773926
Epoch  110  loss  0.3967247947957263 correct 50 Time for this one epoch:  0.09654092788696289
Epoch  120  loss  0.9508948207663475 correct 50 Time for this one epoch:  0.09723901748657227
Epoch  130  loss  0.8998955242526493 correct 49 Time for this one epoch:  0.09580373764038086
Epoch  140  loss  0.005078776429279863 correct 49 Time for this one epoch:  0.09630012512207031
Epoch  150  loss  0.8795986269271945 correct 50 Time for this one epoch:  0.17276716232299805
Epoch  160  loss  0.5005996706128932 correct 49 Time for this one epoch:  0.09817385673522949
Epoch  170  loss  0.036807453050860396 correct 50 Time for this one epoch:  0.11747908592224121
Epoch  180  loss  0.6884735320970135 correct 50 Time for this one epoch:  0.11567401885986328
Epoch  190  loss  0.2846283163284336 correct 49 Time for this one epoch:  0.11926889419555664
Epoch  200  loss  0.5281890095614008 correct 49 Time for this one epoch:  0.09684991836547852
Epoch  210  loss  0.22211814038646976 correct 50 Time for this one epoch:  0.11082100868225098
Epoch  220  loss  0.0625855945166832 correct 50 Time for this one epoch:  0.1533346176147461
Epoch  230  loss  0.673636370639214 correct 50 Time for this one epoch:  0.12077903747558594
Epoch  240  loss  0.18569969317409302 correct 50 Time for this one epoch:  0.10343289375305176
Epoch  250  loss  0.6929716674437645 correct 50 Time for this one epoch:  0.10834884643554688
Epoch  260  loss  0.11479492335691556 correct 50 Time for this one epoch:  0.10184884071350098
Epoch  270  loss  0.37898449029973025 correct 50 Time for this one epoch:  0.09942770004272461
Epoch  280  loss  0.09778643967196993 correct 50 Time for this one epoch:  0.10407376289367676
Epoch  290  loss  0.009366764201578316 correct 50 Time for this one epoch:  0.09991908073425293
Epoch  300  loss  0.13737481438221882 correct 50 Time for this one epoch:  0.11017203330993652
Epoch  310  loss  0.08614131862262189 correct 50 Time for this one epoch:  0.09520220756530762
Epoch  320  loss  0.24539765753717746 correct 50 Time for this one epoch:  0.09487223625183105
Epoch  330  loss  0.1193532699344516 correct 50 Time for this one epoch:  0.09556913375854492
Epoch  340  loss  0.1903557966255539 correct 50 Time for this one epoch:  0.09594225883483887
Epoch  350  loss  0.09397663165116932 correct 50 Time for this one epoch:  0.09506082534790039
Epoch  360  loss  0.15722571902142865 correct 50 Time for this one epoch:  0.09578800201416016
Epoch  370  loss  0.030650233166107183 correct 50 Time for this one epoch:  0.09532499313354492
Epoch  380  loss  0.039295993950709915 correct 50 Time for this one epoch:  0.09584784507751465
Epoch  390  loss  0.7342472063514828 correct 50 Time for this one epoch:  0.09562873840332031
Epoch  400  loss  0.005708149981616204 correct 50 Time for this one epoch:  0.09585380554199219
Epoch  410  loss  0.1618352242492556 correct 50 Time for this one epoch:  0.09674286842346191
Epoch  420  loss  0.2987032718567892 correct 50 Time for this one epoch:  0.09651494026184082
Epoch  430  loss  0.03249866408408001 correct 50 Time for this one epoch:  0.09531688690185547
Epoch  440  loss  0.584005961585173 correct 50 Time for this one epoch:  0.09505319595336914
Epoch  450  loss  0.04444668515494208 correct 50 Time for this one epoch:  0.0961000919342041
Epoch  460  loss  0.005263098591653054 correct 50 Time for this one epoch:  0.09584593772888184
Epoch  470  loss  0.16394242567782474 correct 50 Time for this one epoch:  0.09621906280517578
Epoch  480  loss  0.1950386338755959 correct 50 Time for this one epoch:  0.09717798233032227
Epoch  490  loss  0.02455247754733183 correct 50 Time for this one epoch:  0.09698295593261719
Average time per epoch 0.1316818437576294 (for 500 epochs)

*************************************************************************************************************************************************************

**CPU SPLIT 100**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  0  loss  9.72505060837155 correct 31 Time for this one epoch:  15.064620018005371
Epoch  10  loss  6.861733448874855 correct 37 Time for this one epoch:  0.10435819625854492
Epoch  20  loss  6.21809260664078 correct 43 Time for this one epoch:  0.10008597373962402
Epoch  30  loss  4.6188976633363605 correct 46 Time for this one epoch:  0.09760594367980957
Epoch  40  loss  5.146108567029751 correct 45 Time for this one epoch:  0.09790325164794922
Epoch  50  loss  2.7499628609004336 correct 46 Time for this one epoch:  0.09713196754455566
Epoch  60  loss  3.648447232822579 correct 47 Time for this one epoch:  0.0986928939819336
Epoch  70  loss  3.533420717296875 correct 49 Time for this one epoch:  0.09732794761657715
Epoch  80  loss  2.2670494191895774 correct 50 Time for this one epoch:  0.0977323055267334
Epoch  90  loss  2.473532749253093 correct 45 Time for this one epoch:  0.09763789176940918
Epoch  100  loss  1.5730673083744213 correct 47 Time for this one epoch:  0.09923815727233887
Epoch  110  loss  2.865304534674681 correct 46 Time for this one epoch:  0.09671878814697266
Epoch  120  loss  2.084280119976132 correct 50 Time for this one epoch:  0.09713983535766602
Epoch  130  loss  0.8295251916299007 correct 49 Time for this one epoch:  0.09748625755310059
Epoch  140  loss  1.8293715687096455 correct 48 Time for this one epoch:  0.09687018394470215
Epoch  150  loss  1.1331398479645873 correct 50 Time for this one epoch:  0.17884492874145508
Epoch  160  loss  0.9940082660166294 correct 49 Time for this one epoch:  0.09699892997741699
Epoch  170  loss  1.4072184529986105 correct 50 Time for this one epoch:  0.12226319313049316
Epoch  180  loss  0.47659028474736786 correct 50 Time for this one epoch:  0.0958559513092041
Epoch  190  loss  0.8960664554791635 correct 50 Time for this one epoch:  0.09709000587463379
Epoch  200  loss  1.4474975971077562 correct 50 Time for this one epoch:  0.09662008285522461
Epoch  210  loss  0.30805885057122867 correct 50 Time for this one epoch:  0.09866690635681152
Epoch  220  loss  1.093492122949596 correct 50 Time for this one epoch:  0.09650683403015137
Epoch  230  loss  1.1696791665407462 correct 49 Time for this one epoch:  0.10176301002502441
Epoch  240  loss  1.1146389173483036 correct 49 Time for this one epoch:  0.09838080406188965
Epoch  250  loss  0.5550019416593904 correct 49 Time for this one epoch:  0.09863090515136719
Epoch  260  loss  0.6865303836398186 correct 50 Time for this one epoch:  0.09684300422668457
Epoch  270  loss  0.7301835846480941 correct 50 Time for this one epoch:  0.09818482398986816
Epoch  280  loss  1.32250650259077 correct 50 Time for this one epoch:  0.10346007347106934
Epoch  290  loss  0.45198675107445585 correct 50 Time for this one epoch:  0.09678387641906738
Epoch  300  loss  0.3831551412533475 correct 50 Time for this one epoch:  0.09649395942687988
Epoch  310  loss  0.39121770540616335 correct 50 Time for this one epoch:  0.09910297393798828
Epoch  320  loss  0.4550070768873117 correct 50 Time for this one epoch:  0.10340189933776855
Epoch  330  loss  0.5074081370301559 correct 50 Time for this one epoch:  0.09911584854125977
Epoch  340  loss  1.1828062215481199 correct 50 Time for this one epoch:  0.09651803970336914
Epoch  350  loss  0.30261626369865535 correct 50 Time for this one epoch:  0.1022951602935791
Epoch  360  loss  0.38940217334197 correct 50 Time for this one epoch:  0.09657502174377441
Epoch  370  loss  0.19845431195264804 correct 50 Time for this one epoch:  0.09676694869995117
Epoch  380  loss  0.3436724308592132 correct 50 Time for this one epoch:  0.10119175910949707
Epoch  390  loss  0.2547146599494345 correct 50 Time for this one epoch:  0.0981130599975586
Epoch  400  loss  1.3238240633676803 correct 50 Time for this one epoch:  0.1030721664428711
Epoch  410  loss  0.42909734928028254 correct 50 Time for this one epoch:  0.0969390869140625
Epoch  420  loss  0.7460178951699831 correct 50 Time for this one epoch:  0.09753203392028809
Epoch  430  loss  0.3216103114809664 correct 50 Time for this one epoch:  0.09780192375183105
Epoch  440  loss  0.3772852248332517 correct 50 Time for this one epoch:  0.09668898582458496
Epoch  450  loss  0.08851643119452988 correct 49 Time for this one epoch:  0.0984807014465332
Epoch  460  loss  0.0818211782746482 correct 50 Time for this one epoch:  0.10025501251220703
Epoch  470  loss  0.21681628770018216 correct 50 Time for this one epoch:  0.09874916076660156
Epoch  480  loss  0.3234624189001366 correct 50 Time for this one epoch:  0.0971670150756836
Epoch  490  loss  0.5278458356669552 correct 50 Time for this one epoch:  0.09699106216430664
Average time per epoch 0.12893980836868285 (for 500 epochs)

*************************************************************************************************************************************************************

**CPU DIAG 100**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET diag --RATE 0.05

Epoch  0  loss  2.9867625915935956 correct 43 Time for this one epoch:  14.483262062072754
Epoch  10  loss  1.0679625658336092 correct 45 Time for this one epoch:  0.11416792869567871
Epoch  20  loss  2.0431203404091134 correct 48 Time for this one epoch:  0.0970146656036377
Epoch  30  loss  1.7876704586321237 correct 47 Time for this one epoch:  0.0976419448852539
Epoch  40  loss  1.790504967562086 correct 47 Time for this one epoch:  0.10029792785644531
Epoch  50  loss  1.4812615439284909 correct 49 Time for this one epoch:  0.09689617156982422
Epoch  60  loss  1.4845637986578497 correct 50 Time for this one epoch:  0.09895181655883789
Epoch  70  loss  1.0206502128456156 correct 49 Time for this one epoch:  0.09576177597045898
Epoch  80  loss  0.7234387645850747 correct 49 Time for this one epoch:  0.09654688835144043
Epoch  90  loss  0.14105245237869765 correct 49 Time for this one epoch:  0.09683489799499512
Epoch  100  loss  1.2856819969133524 correct 50 Time for this one epoch:  0.10035490989685059
Epoch  110  loss  1.048657767042506 correct 50 Time for this one epoch:  0.09714603424072266
Epoch  120  loss  1.1687241270189366 correct 50 Time for this one epoch:  0.09662628173828125
Epoch  130  loss  0.5502619832702111 correct 50 Time for this one epoch:  0.10421323776245117
Epoch  140  loss  0.24702976452704523 correct 50 Time for this one epoch:  0.09740805625915527
Epoch  150  loss  0.0391807909990296 correct 50 Time for this one epoch:  0.1821138858795166
Epoch  160  loss  1.0333276847812913 correct 50 Time for this one epoch:  0.09674715995788574
Epoch  170  loss  0.5107388807545739 correct 50 Time for this one epoch:  0.09644627571105957
Epoch  180  loss  0.12367019541546881 correct 50 Time for this one epoch:  0.10601615905761719
Epoch  190  loss  0.16644643658535052 correct 50 Time for this one epoch:  0.09855890274047852
Epoch  200  loss  0.18191683064446126 correct 50 Time for this one epoch:  0.11143994331359863
Epoch  210  loss  0.3866083292673616 correct 50 Time for this one epoch:  0.09699296951293945
Epoch  220  loss  0.38634266926629507 correct 50 Time for this one epoch:  0.09862709045410156
Epoch  230  loss  0.6887253002396043 correct 50 Time for this one epoch:  0.09996414184570312
Epoch  240  loss  0.552800171465727 correct 50 Time for this one epoch:  0.0985560417175293
Epoch  250  loss  0.5178825959083577 correct 50 Time for this one epoch:  0.09692692756652832
Epoch  260  loss  0.1534640602085689 correct 50 Time for this one epoch:  0.09646296501159668
Epoch  270  loss  0.23099154001721664 correct 50 Time for this one epoch:  0.09736514091491699
Epoch  280  loss  0.23610649942302658 correct 50 Time for this one epoch:  0.1580822467803955
Epoch  290  loss  0.05860830625475526 correct 50 Time for this one epoch:  0.09723496437072754
Epoch  300  loss  0.42315136433396466 correct 50 Time for this one epoch:  0.09824204444885254
Epoch  310  loss  0.11861552441750485 correct 50 Time for this one epoch:  0.09704184532165527
Epoch  320  loss  0.29051048579417954 correct 50 Time for this one epoch:  0.0975499153137207
Epoch  330  loss  0.31514809718474857 correct 50 Time for this one epoch:  0.09738492965698242
Epoch  340  loss  0.0018052427059153456 correct 50 Time for this one epoch:  0.09769701957702637
Epoch  350  loss  0.16262077902748145 correct 50 Time for this one epoch:  0.09679365158081055
Epoch  360  loss  0.10208544210302403 correct 50 Time for this one epoch:  0.0968787670135498
Epoch  370  loss  0.03284360655725434 correct 50 Time for this one epoch:  0.10047292709350586
Epoch  380  loss  0.3882733674382164 correct 50 Time for this one epoch:  0.09676218032836914
Epoch  390  loss  0.41040875399387183 correct 50 Time for this one epoch:  0.09772682189941406
Epoch  400  loss  0.16356238922170072 correct 50 Time for this one epoch:  0.1001729965209961
Epoch  410  loss  0.0008024940849185152 correct 50 Time for this one epoch:  0.09713506698608398
Epoch  420  loss  0.15262321400952011 correct 50 Time for this one epoch:  0.09740900993347168
Epoch  430  loss  0.012087328651877716 correct 50 Time for this one epoch:  0.11192584037780762
Epoch  440  loss  0.26028003210817713 correct 50 Time for this one epoch:  0.09894609451293945
Epoch  450  loss  0.13405401596149796 correct 50 Time for this one epoch:  0.09676909446716309
Epoch  460  loss  0.12940293762422497 correct 50 Time for this one epoch:  0.09679269790649414
Epoch  470  loss  0.40395540150697345 correct 50 Time for this one epoch:  0.09761476516723633
Epoch  480  loss  0.1372598254847204 correct 50 Time for this one epoch:  0.09845399856567383
Epoch  490  loss  0.18536791476024017 correct 50 Time for this one epoch:  0.09679293632507324
Average time per epoch 0.12986664247512816 (for 500 epochs)


*************************************************************************************************************************************************************

**CPU XOR 100**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  0  loss  6.53585411127546 correct 32 Time for this one epoch:  14.407131910324097
Epoch  10  loss  5.603335793217767 correct 42 Time for this one epoch:  0.09973001480102539
Epoch  20  loss  6.038310435511155 correct 40 Time for this one epoch:  0.09730219841003418
Epoch  30  loss  4.480140016763446 correct 44 Time for this one epoch:  0.1281740665435791
Epoch  40  loss  2.381396799385879 correct 45 Time for this one epoch:  0.11316108703613281
Epoch  50  loss  3.061331798006092 correct 42 Time for this one epoch:  0.12825226783752441
Epoch  60  loss  3.6561167040976237 correct 48 Time for this one epoch:  0.12523508071899414
Epoch  70  loss  3.6645520516793226 correct 45 Time for this one epoch:  0.09899425506591797
Epoch  80  loss  3.525587043571009 correct 43 Time for this one epoch:  0.10227417945861816
Epoch  90  loss  3.1799198162460653 correct 43 Time for this one epoch:  0.09721088409423828
Epoch  100  loss  1.814971288365723 correct 45 Time for this one epoch:  0.09733009338378906
Epoch  110  loss  1.6188468411974424 correct 50 Time for this one epoch:  0.09722495079040527
Epoch  120  loss  2.189669791345305 correct 47 Time for this one epoch:  0.10094904899597168
Epoch  130  loss  1.5424662349349776 correct 46 Time for this one epoch:  0.09758687019348145
Epoch  140  loss  2.0512669500416187 correct 47 Time for this one epoch:  0.09731769561767578
Epoch  150  loss  0.9776285192686965 correct 50 Time for this one epoch:  0.17271709442138672
Epoch  160  loss  1.7901376079351954 correct 50 Time for this one epoch:  0.09827184677124023
Epoch  170  loss  1.328006802696073 correct 49 Time for this one epoch:  0.09924578666687012
Epoch  180  loss  1.0029120736068045 correct 48 Time for this one epoch:  0.09924697875976562
Epoch  190  loss  1.5752706294654464 correct 49 Time for this one epoch:  0.10056710243225098
Epoch  200  loss  0.9687321331680103 correct 50 Time for this one epoch:  0.09915590286254883
Epoch  210  loss  0.7310861272048952 correct 48 Time for this one epoch:  0.10037612915039062
Epoch  220  loss  1.6551771203200651 correct 48 Time for this one epoch:  0.10016798973083496
Epoch  230  loss  0.7725996655114027 correct 49 Time for this one epoch:  0.09982991218566895
Epoch  240  loss  0.31908796663117633 correct 48 Time for this one epoch:  0.10741496086120605
Epoch  250  loss  0.21088627169034396 correct 50 Time for this one epoch:  0.11066699028015137
Epoch  260  loss  1.2201787275960736 correct 49 Time for this one epoch:  0.1165001392364502
Epoch  270  loss  0.37695115986780026 correct 50 Time for this one epoch:  0.11124300956726074
Epoch  280  loss  0.7194778269151031 correct 50 Time for this one epoch:  0.09766721725463867
Epoch  290  loss  0.6322746903089828 correct 50 Time for this one epoch:  0.10575199127197266
Epoch  300  loss  0.9342291838290461 correct 50 Time for this one epoch:  0.1025390625
Epoch  310  loss  0.5428253536617544 correct 50 Time for this one epoch:  0.16744613647460938
Epoch  320  loss  0.8319143577875762 correct 50 Time for this one epoch:  0.10372233390808105
Epoch  330  loss  1.2194476230316937 correct 50 Time for this one epoch:  0.15498685836791992
Epoch  340  loss  0.34344551434121096 correct 50 Time for this one epoch:  0.13479208946228027
Epoch  350  loss  0.48707989018549647 correct 50 Time for this one epoch:  0.12926888465881348
Epoch  360  loss  0.6355237710900198 correct 50 Time for this one epoch:  0.11125802993774414
Epoch  370  loss  0.7067285428530528 correct 50 Time for this one epoch:  0.1030418872833252
Epoch  380  loss  0.14427690579377123 correct 50 Time for this one epoch:  0.10655808448791504
Epoch  390  loss  0.8939910104413299 correct 50 Time for this one epoch:  0.10548281669616699
Epoch  400  loss  0.5229291657027744 correct 50 Time for this one epoch:  0.10526800155639648
Epoch  410  loss  0.5619311304734258 correct 50 Time for this one epoch:  0.14218425750732422
Epoch  420  loss  0.586446937878757 correct 50 Time for this one epoch:  0.11946487426757812
Epoch  430  loss  0.4851723649132743 correct 50 Time for this one epoch:  0.10039091110229492
Epoch  440  loss  0.6744472581405668 correct 50 Time for this one epoch:  0.09714794158935547
Epoch  450  loss  0.4677447073788017 correct 50 Time for this one epoch:  0.0972743034362793
Epoch  460  loss  0.4290988717367831 correct 50 Time for this one epoch:  0.09790325164794922
Epoch  470  loss  0.11594200383289902 correct 50 Time for this one epoch:  0.1005411148071289
Epoch  480  loss  0.20613663600126372 correct 50 Time for this one epoch:  0.09844684600830078
Epoch  490  loss  0.2045258014505426 correct 50 Time for this one epoch:  0.09927105903625488
Average time per epoch 0.13730681896209718 (for 500 epochs)


*************************************************************************************************************************************************************

**CPU SPLIT 200**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05

Epoch  0  loss  3.0553291809170795 correct 35 Time for this one epoch:  16.143571853637695
Epoch  10  loss  1.9877204612303871 correct 45 Time for this one epoch:  0.17695307731628418
Epoch  20  loss  1.394919468663192 correct 43 Time for this one epoch:  0.16347575187683105
Epoch  30  loss  1.660366197634733 correct 48 Time for this one epoch:  0.1585557460784912
Epoch  40  loss  2.194611107128106 correct 48 Time for this one epoch:  0.15940022468566895
Epoch  50  loss  0.7590661008526589 correct 48 Time for this one epoch:  0.16132497787475586
Epoch  60  loss  1.189625541441177 correct 50 Time for this one epoch:  0.16046619415283203
Epoch  70  loss  0.3127225271235174 correct 48 Time for this one epoch:  0.16010713577270508
Epoch  80  loss  1.2531288726776026 correct 48 Time for this one epoch:  0.1600189208984375
Epoch  90  loss  1.5978705493853331 correct 50 Time for this one epoch:  0.15758109092712402
Epoch  100  loss  0.5257421564795048 correct 48 Time for this one epoch:  0.1595468521118164
Epoch  110  loss  1.3230143206719671 correct 49 Time for this one epoch:  0.15815281867980957
Epoch  120  loss  0.6768214675497484 correct 48 Time for this one epoch:  0.16037225723266602
Epoch  130  loss  1.528170788934073 correct 48 Time for this one epoch:  0.1615431308746338
Epoch  140  loss  1.065196586092086 correct 48 Time for this one epoch:  0.16486310958862305
Epoch  150  loss  0.37985033422329373 correct 50 Time for this one epoch:  0.23757290840148926
Epoch  160  loss  0.5872916915636364 correct 50 Time for this one epoch:  0.16250085830688477
Epoch  170  loss  0.39959515369046034 correct 48 Time for this one epoch:  0.1603221893310547
Epoch  180  loss  0.23115883820074135 correct 50 Time for this one epoch:  0.16001105308532715
Epoch  190  loss  0.28053368760145864 correct 50 Time for this one epoch:  0.16314172744750977
Epoch  200  loss  0.18661142788160986 correct 50 Time for this one epoch:  0.16205811500549316
Epoch  210  loss  0.2541105484115438 correct 50 Time for this one epoch:  0.1583700180053711
Epoch  220  loss  0.3570963260466817 correct 50 Time for this one epoch:  0.16591215133666992
Epoch  230  loss  0.14179729048106082 correct 50 Time for this one epoch:  0.16232085227966309
Epoch  240  loss  0.12036868698401766 correct 50 Time for this one epoch:  0.16121816635131836
Epoch  250  loss  0.3066312745452754 correct 50 Time for this one epoch:  0.15971899032592773
Epoch  260  loss  0.16724595923374583 correct 50 Time for this one epoch:  0.15868592262268066
Epoch  270  loss  0.124573608580636 correct 50 Time for this one epoch:  0.166884183883667
Epoch  280  loss  0.28215387133748643 correct 50 Time for this one epoch:  0.1596369743347168
Epoch  290  loss  0.1605915740343693 correct 50 Time for this one epoch:  0.16039299964904785
Epoch  300  loss  0.19478491002668874 correct 50 Time for this one epoch:  0.16332006454467773
Epoch  310  loss  0.298354256477226 correct 50 Time for this one epoch:  0.1596212387084961
Epoch  320  loss  0.0984785615636479 correct 50 Time for this one epoch:  0.15825581550598145
Epoch  330  loss  0.13958303215922613 correct 50 Time for this one epoch:  0.15946173667907715
Epoch  340  loss  0.055401540607896024 correct 50 Time for this one epoch:  0.1592702865600586
Epoch  350  loss  0.1774531428992907 correct 50 Time for this one epoch:  0.17219305038452148
Epoch  360  loss  0.09081872362431043 correct 50 Time for this one epoch:  0.15954113006591797
Epoch  370  loss  0.10692940561222863 correct 50 Time for this one epoch:  0.15853095054626465
Epoch  380  loss  0.023541931054619686 correct 50 Time for this one epoch:  0.1617870330810547
Epoch  390  loss  0.34582341284928175 correct 50 Time for this one epoch:  0.15962910652160645
Epoch  400  loss  0.0908001859454005 correct 50 Time for this one epoch:  0.15975499153137207
Epoch  410  loss  0.17764879586063434 correct 50 Time for this one epoch:  0.17721104621887207
Epoch  420  loss  0.12764693003260175 correct 50 Time for this one epoch:  0.15792393684387207
Epoch  430  loss  0.16096456095874923 correct 50 Time for this one epoch:  0.1593029499053955
Epoch  440  loss  0.24387498617714748 correct 50 Time for this one epoch:  0.1599440574645996
Epoch  450  loss  0.18593141355849047 correct 50 Time for this one epoch:  0.1590571403503418
Epoch  460  loss  0.2380341129980869 correct 50 Time for this one epoch:  0.15928387641906738
Epoch  470  loss  0.06937772662845267 correct 50 Time for this one epoch:  0.1630420684814453
Epoch  480  loss  0.17336973068772443 correct 50 Time for this one epoch:  0.17708206176757812
Epoch  490  loss  0.06308015279026989 correct 50 Time for this one epoch:  0.15737414360046387
Average time per epoch 0.19674319791793823 (for 500 epochs)


*************************************************************************************************************************************************************

**GPU SIMPLE 100**


! python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch  0  loss  5.484306171892152 correct 40 Time for this one epoch:  6.5082175731658936
Epoch  10  loss  2.9323089544102734 correct 47 Time for this one epoch:  3.1894373893737793
Epoch  20  loss  2.413688367950413 correct 48 Time for this one epoch:  3.2654995918273926
Epoch  30  loss  3.600289647974454 correct 48 Time for this one epoch:  3.1660397052764893
Epoch  40  loss  2.3699523868654593 correct 49 Time for this one epoch:  3.2373249530792236
Epoch  50  loss  1.6904989557030372 correct 50 Time for this one epoch:  3.002399444580078
Epoch  60  loss  0.26970675061368665 correct 48 Time for this one epoch:  2.8354456424713135
Epoch  70  loss  0.2655866370303471 correct 50 Time for this one epoch:  2.682662010192871
Epoch  80  loss  0.4133807580871496 correct 49 Time for this one epoch:  2.3650636672973633
Epoch  90  loss  1.8623147691714792 correct 50 Time for this one epoch:  2.0501320362091064
Epoch  100  loss  1.606747170646298 correct 48 Time for this one epoch:  1.8368020057678223
Epoch  110  loss  1.1986817026524563 correct 50 Time for this one epoch:  1.8257808685302734
Epoch  120  loss  2.0559678533226045 correct 50 Time for this one epoch:  1.821211576461792
Epoch  130  loss  0.23327261483155703 correct 49 Time for this one epoch:  1.8313171863555908
Epoch  140  loss  0.5005269906566464 correct 48 Time for this one epoch:  1.813776969909668
Epoch  150  loss  0.40258817207228265 correct 48 Time for this one epoch:  1.8330397605895996
Epoch  160  loss  0.7098849273148673 correct 47 Time for this one epoch:  1.8100435733795166
Epoch  170  loss  0.9112553547061945 correct 49 Time for this one epoch:  1.8037030696868896
Epoch  180  loss  0.23084716688066995 correct 50 Time for this one epoch:  1.826547622680664
Epoch  190  loss  0.49828435040346514 correct 49 Time for this one epoch:  1.810915231704712
Epoch  200  loss  0.5813669750877584 correct 50 Time for this one epoch:  1.8870937824249268
Epoch  210  loss  0.2908332099769826 correct 50 Time for this one epoch:  1.8226923942565918
Epoch  220  loss  0.46890114437647407 correct 50 Time for this one epoch:  1.8474946022033691
Epoch  230  loss  0.13378940610565979 correct 50 Time for this one epoch:  1.8283751010894775
Epoch  240  loss  0.7203080552338006 correct 50 Time for this one epoch:  1.8279802799224854
Epoch  250  loss  0.9152361540487677 correct 50 Time for this one epoch:  2.3623926639556885
Epoch  260  loss  0.25788159485132145 correct 50 Time for this one epoch:  2.7185418605804443
Epoch  270  loss  0.18502777529120304 correct 50 Time for this one epoch:  1.8380732536315918
Epoch  280  loss  0.01580002966230245 correct 50 Time for this one epoch:  1.8064236640930176
Epoch  290  loss  0.6329240733945416 correct 50 Time for this one epoch:  1.8816466331481934
Epoch  300  loss  0.3831287998529687 correct 50 Time for this one epoch:  1.9494190216064453
Epoch  310  loss  0.06580891171164824 correct 50 Time for this one epoch:  2.288874864578247
Epoch  320  loss  0.40984380153473626 correct 50 Time for this one epoch:  2.3958659172058105
Epoch  330  loss  0.6778645652884526 correct 50 Time for this one epoch:  2.508524179458618
Epoch  340  loss  0.09110794734994143 correct 50 Time for this one epoch:  2.8516006469726562
Epoch  350  loss  0.19822831095549148 correct 50 Time for this one epoch:  2.934384822845459
Epoch  360  loss  0.4456444007674505 correct 50 Time for this one epoch:  3.1250200271606445
Epoch  370  loss  0.555017881714616 correct 50 Time for this one epoch:  3.1880385875701904
Epoch  380  loss  0.24717876702394032 correct 50 Time for this one epoch:  3.256791591644287
Epoch  390  loss  0.6704384090422993 correct 49 Time for this one epoch:  3.1891603469848633
Epoch  400  loss  0.0005166346888598802 correct 50 Time for this one epoch:  3.4399197101593018
Epoch  410  loss  0.00459889725981705 correct 50 Time for this one epoch:  3.1961567401885986
Epoch  420  loss  0.4907274051439088 correct 50 Time for this one epoch:  3.2226765155792236
Epoch  430  loss  0.0242137130051222 correct 50 Time for this one epoch:  3.1640524864196777
Epoch  440  loss  0.710255076448525 correct 50 Time for this one epoch:  3.2125837802886963
Epoch  450  loss  0.0006948867124830725 correct 50 Time for this one epoch:  3.218937635421753
Epoch  460  loss  0.31552618132881055 correct 50 Time for this one epoch:  3.289210557937622
Epoch  470  loss  0.04550679249120133 correct 50 Time for this one epoch:  2.669544219970703
Epoch  480  loss  0.38643779488783375 correct 50 Time for this one epoch:  1.9712550640106201
Epoch  490  loss  0.22915256780586502 correct 50 Time for this one epoch:  1.9011528491973877
Average time per epoch 2.4665619530677794 (for 500 epochs)


*************************************************************************************************************************************************************

**GPU SPLIT 100**


! python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch  0  loss  5.840771601280058 correct 31 Time for this one epoch:  8.03837776184082
Epoch  10  loss  6.021706769681062 correct 43 Time for this one epoch:  2.6056981086730957
Epoch  20  loss  3.961419679800804 correct 45 Time for this one epoch:  1.9753568172454834
Epoch  30  loss  3.7758043988095995 correct 47 Time for this one epoch:  1.8420772552490234
Epoch  40  loss  2.304424355110124 correct 48 Time for this one epoch:  1.8484716415405273
Epoch  50  loss  1.351879153776845 correct 48 Time for this one epoch:  1.8980727195739746
Epoch  60  loss  1.9831718231604503 correct 49 Time for this one epoch:  1.8434839248657227
Epoch  70  loss  1.9484507792528283 correct 49 Time for this one epoch:  1.8323299884796143
Epoch  80  loss  0.937100996751668 correct 47 Time for this one epoch:  1.8349502086639404
Epoch  90  loss  1.3157047167081903 correct 49 Time for this one epoch:  1.8628830909729004
Epoch  100  loss  1.3502153966143504 correct 50 Time for this one epoch:  1.8730206489562988
Epoch  110  loss  0.9614622009250089 correct 50 Time for this one epoch:  1.8516592979431152
Epoch  120  loss  1.4824921294605484 correct 50 Time for this one epoch:  1.9149730205535889
Epoch  130  loss  0.4055779769379145 correct 50 Time for this one epoch:  1.8486030101776123
Epoch  140  loss  0.5908994330111034 correct 49 Time for this one epoch:  1.853196144104004
Epoch  150  loss  0.2692536841606813 correct 49 Time for this one epoch:  1.8998651504516602
Epoch  160  loss  0.8159461757778975 correct 50 Time for this one epoch:  1.8432340621948242
Epoch  170  loss  1.5148969306817794 correct 49 Time for this one epoch:  1.9918017387390137
Epoch  180  loss  1.4210096625843318 correct 49 Time for this one epoch:  2.319254159927368
Epoch  190  loss  0.2611112569820011 correct 49 Time for this one epoch:  2.6520214080810547
Epoch  200  loss  0.3457786752070261 correct 49 Time for this one epoch:  2.8348329067230225
Epoch  210  loss  1.7160667252338486 correct 49 Time for this one epoch:  2.834286689758301
Epoch  220  loss  1.2883783468002417 correct 50 Time for this one epoch:  2.955580472946167
Epoch  230  loss  2.0792148259949705 correct 48 Time for this one epoch:  3.1751492023468018
Epoch  240  loss  0.05842958095492932 correct 50 Time for this one epoch:  3.2637338638305664
Epoch  250  loss  0.453379583579902 correct 48 Time for this one epoch:  3.247481346130371
Epoch  260  loss  0.33233853822572385 correct 49 Time for this one epoch:  3.259718656539917
Epoch  270  loss  1.319343821994691 correct 49 Time for this one epoch:  3.2634668350219727
Epoch  280  loss  0.9184869859588466 correct 50 Time for this one epoch:  3.3649206161499023
Epoch  290  loss  1.193026911372132 correct 49 Time for this one epoch:  1.8653740882873535
Epoch  300  loss  0.318470259682409 correct 49 Time for this one epoch:  1.8434438705444336
Epoch  310  loss  0.13467516216651895 correct 50 Time for this one epoch:  1.8847754001617432
Epoch  320  loss  0.5463988471510445 correct 49 Time for this one epoch:  1.8462600708007812
Epoch  330  loss  0.11132272712318458 correct 48 Time for this one epoch:  1.8768227100372314
Epoch  340  loss  0.5329246611674497 correct 49 Time for this one epoch:  1.8649914264678955
Epoch  350  loss  0.48297088783707187 correct 50 Time for this one epoch:  1.8618009090423584
Epoch  360  loss  0.19427304705502693 correct 50 Time for this one epoch:  1.8812072277069092
Epoch  370  loss  0.4241002741042984 correct 49 Time for this one epoch:  1.831787347793579
Epoch  380  loss  0.8732552788893708 correct 49 Time for this one epoch:  1.9714670181274414
Epoch  390  loss  0.11470863306189702 correct 49 Time for this one epoch:  3.215519428253174
Epoch  400  loss  0.07771359922945818 correct 50 Time for this one epoch:  3.1661009788513184
Epoch  410  loss  0.23716177710176814 correct 49 Time for this one epoch:  3.1940629482269287
Epoch  420  loss  1.0276425108970484 correct 50 Time for this one epoch:  3.223900318145752
Epoch  430  loss  0.15336938970802916 correct 50 Time for this one epoch:  3.2284862995147705
Epoch  440  loss  0.35698972356189773 correct 50 Time for this one epoch:  3.2215898036956787
Epoch  450  loss  0.28383786362492297 correct 49 Time for this one epoch:  3.2443766593933105
Epoch  460  loss  0.587798740808503 correct 50 Time for this one epoch:  3.206395387649536
Epoch  470  loss  0.24112058736707198 correct 49 Time for this one epoch:  3.1944918632507324
Epoch  480  loss  1.3534967013525727 correct 48 Time for this one epoch:  2.887373208999634
Epoch  490  loss  0.8518423584668231 correct 49 Time for this one epoch:  2.479937791824341
Average time per epoch 2.501371241092682 (for 500 epochs)


*************************************************************************************************************************************************************

**GPU DIAG 100**


! python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET diag --RATE 0.05

Epoch  0  loss  3.906900599160317 correct 44 Time for this one epoch:  4.583714723587036
Epoch  10  loss  0.5410602653686136 correct 47 Time for this one epoch:  2.3477096557617188
Epoch  20  loss  1.1867552148663092 correct 49 Time for this one epoch:  2.6591544151306152
Epoch  30  loss  0.4063210402316006 correct 49 Time for this one epoch:  2.873408317565918
Epoch  40  loss  0.4776010655706419 correct 49 Time for this one epoch:  3.2010281085968018
Epoch  50  loss  0.24456069997268104 correct 50 Time for this one epoch:  3.2986855506896973
Epoch  60  loss  0.38535646754668146 correct 49 Time for this one epoch:  3.362196207046509
Epoch  70  loss  0.2973351917556839 correct 50 Time for this one epoch:  3.2950425148010254
Epoch  80  loss  1.0004448123797318 correct 50 Time for this one epoch:  3.1491892337799072
Epoch  90  loss  0.18963374509730788 correct 50 Time for this one epoch:  2.394031286239624
Epoch  100  loss  0.6688904778937618 correct 50 Time for this one epoch:  1.843782901763916
Epoch  110  loss  0.1346994689539373 correct 50 Time for this one epoch:  1.9007797241210938
Epoch  120  loss  0.37446599447958184 correct 50 Time for this one epoch:  1.8588011264801025
Epoch  130  loss  0.07647900945753461 correct 50 Time for this one epoch:  1.911644458770752
Epoch  140  loss  0.3865531278334513 correct 50 Time for this one epoch:  1.9710943698883057
Epoch  150  loss  0.04381609507599257 correct 50 Time for this one epoch:  3.576120615005493
Epoch  160  loss  0.3270882089542369 correct 50 Time for this one epoch:  2.5572562217712402
Epoch  170  loss  0.012916967522229978 correct 50 Time for this one epoch:  2.899420976638794
Epoch  180  loss  0.04419773016912935 correct 50 Time for this one epoch:  3.1942975521087646
Epoch  190  loss  0.23172832868587814 correct 50 Time for this one epoch:  3.307408094406128
Epoch  200  loss  0.41231560300912057 correct 50 Time for this one epoch:  3.3547985553741455
Epoch  210  loss  0.050301370960984665 correct 50 Time for this one epoch:  3.523433208465576
Epoch  220  loss  0.01711692151428674 correct 50 Time for this one epoch:  3.478527069091797
Epoch  230  loss  0.01974204383245922 correct 50 Time for this one epoch:  2.5384106636047363
Epoch  240  loss  0.08332937198838285 correct 50 Time for this one epoch:  2.0971314907073975
Epoch  250  loss  0.008165006761369343 correct 50 Time for this one epoch:  2.0258095264434814
Epoch  260  loss  0.001670626598662931 correct 50 Time for this one epoch:  3.0464351177215576
Epoch  270  loss  0.014471133693650754 correct 50 Time for this one epoch:  2.04113507270813
Epoch  280  loss  0.0002027413241418946 correct 50 Time for this one epoch:  2.274421215057373
Epoch  290  loss  2.9148452082397867e-06 correct 50 Time for this one epoch:  2.9536707401275635
Epoch  300  loss  0.006480824462357914 correct 50 Time for this one epoch:  3.0318779945373535
Epoch  310  loss  0.014089689590982345 correct 50 Time for this one epoch:  3.4812686443328857
Epoch  320  loss  0.006213898990885621 correct 50 Time for this one epoch:  3.500823497772217
Epoch  330  loss  0.0032892041903523194 correct 50 Time for this one epoch:  3.4347734451293945
Epoch  340  loss  0.015511388959385562 correct 50 Time for this one epoch:  3.379014492034912
Epoch  350  loss  0.0036113385258091505 correct 50 Time for this one epoch:  3.1886637210845947
Epoch  360  loss  0.008557150169590915 correct 50 Time for this one epoch:  2.1222574710845947
Epoch  370  loss  0.005187610321459091 correct 50 Time for this one epoch:  2.0736918449401855
Epoch  380  loss  0.0008294580937789944 correct 50 Time for this one epoch:  3.301508665084839
Epoch  390  loss  0.0033387801933850367 correct 50 Time for this one epoch:  1.9799861907958984
Epoch  400  loss  0.015321819131444214 correct 50 Time for this one epoch:  1.9672205448150635
Epoch  410  loss  0.0016533777043331384 correct 50 Time for this one epoch:  2.232663869857788
Epoch  420  loss  0.0005877863857869483 correct 50 Time for this one epoch:  2.6429452896118164
Epoch  430  loss  0.007887702576693356 correct 50 Time for this one epoch:  2.9283573627471924
Epoch  440  loss  0.009373728863563129 correct 50 Time for this one epoch:  3.1138174533843994
Epoch  450  loss  0.00023877908551188785 correct 50 Time for this one epoch:  3.698867082595825
Epoch  460  loss  0.0006025496405657556 correct 50 Time for this one epoch:  3.6065008640289307
Epoch  470  loss  0.011565080940405837 correct 50 Time for this one epoch:  3.437886953353882
Epoch  480  loss  0.002601719565349358 correct 50 Time for this one epoch:  3.372060775756836
Epoch  490  loss  0.002249924973426208 correct 50 Time for this one epoch:  3.3206193447113037
Average time per epoch 2.5870858826637266 (for 500 epochs)


*************************************************************************************************************************************************************

**GPU XOR 100**


python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch  0  loss  12.318772028636593 correct 27 Time for this one epoch:  6.356503963470459
Epoch  10  loss  7.7253586719487926 correct 27 Time for this one epoch:  2.563072681427002
Epoch  20  loss  6.327453152499823 correct 29 Time for this one epoch:  1.9557445049285889
Epoch  30  loss  4.214484129590418 correct 32 Time for this one epoch:  1.9310708045959473
Epoch  40  loss  5.839541604614288 correct 35 Time for this one epoch:  1.887805461883545
Epoch  50  loss  4.957271769690683 correct 35 Time for this one epoch:  1.928114652633667
Epoch  60  loss  4.453488162031238 correct 34 Time for this one epoch:  1.9456920623779297
Epoch  70  loss  2.328391962006875 correct 34 Time for this one epoch:  1.957345962524414
Epoch  80  loss  1.017769326991042 correct 36 Time for this one epoch:  2.511348247528076
Epoch  90  loss  1.9672357973575756 correct 44 Time for this one epoch:  2.917484998703003
Epoch  100  loss  0.2995710883188316 correct 42 Time for this one epoch:  3.1169331073760986
Epoch  110  loss  1.5438968453502777 correct 43 Time for this one epoch:  3.372152328491211
Epoch  120  loss  0.35929708135160293 correct 47 Time for this one epoch:  3.37459135055542
Epoch  130  loss  1.6324422625341957 correct 47 Time for this one epoch:  3.267421007156372
Epoch  140  loss  0.05666013403291755 correct 48 Time for this one epoch:  3.3831844329833984
Epoch  150  loss  0.21087796317527768 correct 46 Time for this one epoch:  2.5148203372955322
Epoch  160  loss  10.678770188613276 correct 47 Time for this one epoch:  1.880608081817627
Epoch  170  loss  0.11074007849997301 correct 48 Time for this one epoch:  1.9461843967437744
Epoch  180  loss  0.0714427079960474 correct 48 Time for this one epoch:  1.9679286479949951
Epoch  190  loss  0.03428392288899678 correct 49 Time for this one epoch:  1.9162840843200684
Epoch  200  loss  1.9225987352814995 correct 44 Time for this one epoch:  1.935683012008667
Epoch  210  loss  0.14678468461454786 correct 48 Time for this one epoch:  1.9393560886383057
Epoch  220  loss  0.11329878485994574 correct 49 Time for this one epoch:  2.1066484451293945
Epoch  230  loss  6.785659013476263 correct 44 Time for this one epoch:  2.592478036880493
Epoch  240  loss  0.04658591241908074 correct 48 Time for this one epoch:  2.978461503982544
Epoch  250  loss  0.022322444067711122 correct 49 Time for this one epoch:  3.1024258136749268
Epoch  260  loss  0.08911614245537326 correct 48 Time for this one epoch:  3.275808572769165
Epoch  270  loss  0.08994501319941199 correct 49 Time for this one epoch:  3.4334654808044434
Epoch  280  loss  0.17683017712869817 correct 49 Time for this one epoch:  3.6428327560424805
Epoch  290  loss  0.0338842587970493 correct 49 Time for this one epoch:  1.9716427326202393
Epoch  300  loss  0.03002518089121747 correct 50 Time for this one epoch:  1.9970192909240723
Epoch  310  loss  0.039527223012572534 correct 48 Time for this one epoch:  2.029266119003296
Epoch  320  loss  0.8008851290586403 correct 46 Time for this one epoch:  2.82893705368042
Epoch  330  loss  0.03511162719699581 correct 50 Time for this one epoch:  3.0539498329162598
Epoch  340  loss  0.030848266976935743 correct 50 Time for this one epoch:  3.375472068786621
Epoch  350  loss  0.031487483196663085 correct 50 Time for this one epoch:  3.4148929119110107
Epoch  360  loss  1.719499770690447 correct 40 Time for this one epoch:  3.434166193008423
Epoch  370  loss  0.4639322961082617 correct 41 Time for this one epoch:  3.4951281547546387
Epoch  380  loss  0.1930379421570985 correct 48 Time for this one epoch:  3.3999273777008057
Epoch  390  loss  0.49406762069336807 correct 45 Time for this one epoch:  2.579545021057129
Epoch  400  loss  1.2798393832641348 correct 44 Time for this one epoch:  2.0370264053344727
Epoch  410  loss  0.47144893760936285 correct 45 Time for this one epoch:  1.9856500625610352
Epoch  420  loss  0.23559353928172355 correct 42 Time for this one epoch:  2.0031003952026367
Epoch  430  loss  0.9100582630102428 correct 46 Time for this one epoch:  1.9702081680297852
Epoch  440  loss  0.34241574863922325 correct 47 Time for this one epoch:  3.756026029586792
Epoch  450  loss  0.7275531751672872 correct 48 Time for this one epoch:  2.8987298011779785
Epoch  460  loss  0.5787245449747853 correct 48 Time for this one epoch:  3.0967178344726562
Epoch  470  loss  0.686468461542328 correct 45 Time for this one epoch:  3.345379590988159
Epoch  480  loss  10.065672517947505 correct 48 Time for this one epoch:  3.3334789276123047
Epoch  490  loss  3.0953004778670232 correct 48 Time for this one epoch:  3.3556201457977295
Average time per epoch 2.610798746585846 (for 500 epochs)


*************************************************************************************************************************************************************

**GPU SPLIT 200**


python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05

Epoch  0  loss  2.8581305409238644 correct 42 Time for this one epoch:  4.1383843421936035
Epoch  10  loss  0.44261504024974685 correct 50 Time for this one epoch:  1.9526913166046143
Epoch  20  loss  0.3037313702858554 correct 50 Time for this one epoch:  3.3808906078338623
Epoch  30  loss  0.07050877552892268 correct 50 Time for this one epoch:  3.427255868911743
Epoch  40  loss  0.1880937211226572 correct 50 Time for this one epoch:  3.4104976654052734
Epoch  50  loss  0.04909328552818632 correct 50 Time for this one epoch:  2.3503739833831787
Epoch  60  loss  0.07035201167203149 correct 50 Time for this one epoch:  1.9551129341125488
Epoch  70  loss  0.026875699127429237 correct 50 Time for this one epoch:  1.9401469230651855
Epoch  80  loss  0.11429621562002637 correct 50 Time for this one epoch:  1.9761581420898438
Epoch  90  loss  0.0967993596914981 correct 50 Time for this one epoch:  1.9341461658477783
Epoch  100  loss  0.03071802020809293 correct 50 Time for this one epoch:  1.9441664218902588
Epoch  110  loss  0.03846271783787662 correct 50 Time for this one epoch:  2.506021738052368
Epoch  120  loss  0.08282814565576138 correct 50 Time for this one epoch:  3.3712828159332275
Epoch  130  loss  0.02729436374226556 correct 50 Time for this one epoch:  3.377195119857788
Epoch  140  loss  0.1041648424767555 correct 50 Time for this one epoch:  3.3816375732421875
Epoch  150  loss  0.09221888643343185 correct 50 Time for this one epoch:  2.313232183456421
Epoch  160  loss  0.07199309981161303 correct 50 Time for this one epoch:  1.9921984672546387
Epoch  170  loss  0.017405647476519515 correct 50 Time for this one epoch:  1.9932060241699219
Epoch  180  loss  0.10531972441526548 correct 50 Time for this one epoch:  1.9964048862457275
Epoch  190  loss  0.047730733955402 correct 50 Time for this one epoch:  1.973978042602539
Epoch  200  loss  0.007692980117547587 correct 50 Time for this one epoch:  3.0351805686950684
Epoch  210  loss  0.003901819113255317 correct 50 Time for this one epoch:  3.393001079559326
Epoch  220  loss  0.010807274153589511 correct 50 Time for this one epoch:  3.5254595279693604
Epoch  230  loss  0.01808581421877632 correct 50 Time for this one epoch:  3.3810713291168213
Epoch  240  loss  0.06967944498851 correct 50 Time for this one epoch:  3.2860918045043945
Epoch  250  loss  0.014662858069895836 correct 50 Time for this one epoch:  1.9701895713806152
Epoch  260  loss  0.0003468984680333288 correct 50 Time for this one epoch:  2.091700315475464
Epoch  270  loss  0.04182981939693687 correct 50 Time for this one epoch:  3.002995252609253
Epoch  280  loss  0.08291312819382726 correct 50 Time for this one epoch:  3.483757734298706
Epoch  290  loss  0.021433666164471187 correct 50 Time for this one epoch:  3.4354255199432373
Epoch  300  loss  0.054542340056862676 correct 50 Time for this one epoch:  3.348159074783325
Epoch  310  loss  0.004740986568146441 correct 50 Time for this one epoch:  2.925128936767578
Epoch  320  loss  0.03816074498950911 correct 50 Time for this one epoch:  2.098572254180908
Epoch  330  loss  0.001177758970959833 correct 50 Time for this one epoch:  1.9687514305114746
Epoch  340  loss  0.02843004240581818 correct 50 Time for this one epoch:  1.9874207973480225
Epoch  350  loss  0.008855643846510667 correct 50 Time for this one epoch:  2.0054240226745605
Epoch  360  loss  0.018671246776746534 correct 50 Time for this one epoch:  2.7064976692199707
Epoch  370  loss  0.010163844203040789 correct 50 Time for this one epoch:  3.543607473373413
Epoch  380  loss  0.0013071175018211458 correct 50 Time for this one epoch:  3.173004627227783
Epoch  390  loss  0.031242987350990577 correct 50 Time for this one epoch:  2.0367345809936523
Epoch  400  loss  0.019294759390338475 correct 50 Time for this one epoch:  2.04075288772583
Epoch  410  loss  0.06386326722378823 correct 50 Time for this one epoch:  2.047236204147339
Epoch  420  loss  0.011894539574178065 correct 50 Time for this one epoch:  2.2442948818206787
Epoch  430  loss  0.039800914875026606 correct 50 Time for this one epoch:  3.058905839920044
Epoch  440  loss  0.005888560409218241 correct 50 Time for this one epoch:  3.599670648574829
Epoch  450  loss  0.019161796843595306 correct 50 Time for this one epoch:  3.568805456161499
Epoch  460  loss  0.030717898220937595 correct 50 Time for this one epoch:  2.4324026107788086
Epoch  470  loss  0.01996460395368369 correct 50 Time for this one epoch:  2.050757884979248
Epoch  480  loss  0.00038621545942868805 correct 50 Time for this one epoch:  2.0134215354919434
Epoch  490  loss  0.0019699180087571094 correct 50 Time for this one epoch:  3.275712251663208
Average time per epoch 2.7126361966133117 (for 500 epochs)

#Example: time python runExperiment.py 1 layerFixedExp YES NO
import os
import sys
import numpy
# argments:
# [1] - Device argument
if len(sys.argv) < 5:
    print "Not enough arguments:\n"
    print "DEVICE, EXP_NAME, CLEAN_MODELS, TEST_ONLY\n"
    exit;

DEVICE = sys.argv[1]
EXP_NAME = sys.argv[2]
CLEAN_MODELS = sys.argv[3]
TEST_ONLY = sys.argv[4]

os.system('mkdir ../results/yolo/'+EXP_NAME)
clean_models = 1;
if (CLEAN_MODELS == 'NO'):
    clean_models = 0;

test_only = 0;
if (TEST_ONLY == 'YES'):
    test_only = 1;

exp_folder = '../results/yolo/'+EXP_NAME+'/'

#Helper function to return the best scores from a scores list
def getBestEpoch(scoreFile, output=False, outputFile=''):
    scores = numpy.loadtxt(open(scoreFile, "rb"), delimiter=",", skiprows=1);
    #Choose based on test mAP
    best_epoch_index = numpy.argmax(scores[:,2])
    if (output == True and outputFile != ''):
        best = scores[best_epoch_index,:]
        best = best.reshape(1, best.shape[0])
        numpy.savetxt(exp_folder+outputFile, best, delimiter=',',comments='',fmt='%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', header="Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train")
    return epochs[best_epoch_index];

#all  output to ../results/yolo/exp_name
    #score files: 4x(kfolds), 1x (kfolds average), 1x (trainval_test)
    #best crossval average epoch stats (1 line file)
    #best trainval h5 model (potential best model)
    #image outputs and proposals on all test images (just for qualitative comparison)

epochs = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]

#Train/Val 1
train_val_file1 = exp_folder+'train1_val1_score.csv'
f = open(train_val_file1, 'a+')
f.write('Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train\n')
f.close()
if not test_only:
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python train.py voc_boat_train1')
for epoch in epochs:
    f = open(train_val_file1, 'a+')
    f.write(str(epoch)+',')
    f.close()
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_val1 voc_boat_train1_'+str(epoch)+' '+train_val_file1)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file1+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_train1 voc_boat_train1_'+str(epoch)+' '+train_val_file1)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file1+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    f = open(train_val_file1, 'a+')
    #remove extra comma
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write('\n')
    f.close()
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')

#Train/Val 2
train_val_file2 = exp_folder+'train2_val2_score.csv'
f = open(train_val_file2, 'a+')
f.write('Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train\n')
f.close()
if not test_only:
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python train.py voc_boat_train2')
for epoch in epochs:
    f = open(train_val_file2, 'a+')
    f.write(str(epoch)+',')
    f.close()
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_val2 voc_boat_train2_'+str(epoch)+' '+train_val_file2)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file2+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_train2 voc_boat_train2_'+str(epoch)+' '+train_val_file2)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file2+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    f = open(train_val_file2, 'a+')
    #remove extra comma
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write('\n')
    f.close()
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
          
#Train/Val 3
train_val_file3 = exp_folder+'train3_val3_score.csv'
f = open(train_val_file3, 'a+')
f.write('Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train\n')
f.close()
if not test_only:
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python train.py voc_boat_train3')
for epoch in epochs:
    f = open(train_val_file3, 'a+')
    f.write(str(epoch)+',')
    f.close()
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_val3 voc_boat_train3_'+str(epoch)+' '+train_val_file3)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file3+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_train3 voc_boat_train3_'+str(epoch)+' '+train_val_file3)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file3+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    f = open(train_val_file3, 'a+')
    #remove extra comma
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write('\n')
    f.close()
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
          
#Train/Val 4
train_val_file4 = exp_folder+'train4_val4_score.csv'
f = open(train_val_file4, 'a+')
f.write('Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train\n')
f.close()
if not test_only:
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python train.py voc_boat_train4')
for epoch in epochs:
    f = open(train_val_file4, 'a+')
    f.write(str(epoch)+',')
    f.close()
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_val4 voc_boat_train4_'+str(epoch)+' '+train_val_file4)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file4+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_train4 voc_boat_train4_'+str(epoch)+' '+train_val_file4)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+train_val_file4+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    f = open(train_val_file4, 'a+')
    #remove extra comma
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write('\n')
    f.close()
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')

#Average KFolds results into 1 file
kfolds_score_file = exp_folder+'kfolds_average_score.csv'
tv1 = numpy.loadtxt(open(train_val_file1, "rb"), delimiter=",", skiprows=1);
tv2 = numpy.loadtxt(open(train_val_file2, "rb"), delimiter=",", skiprows=1);
tv3 = numpy.loadtxt(open(train_val_file3, "rb"), delimiter=",", skiprows=1);
tv4 = numpy.loadtxt(open(train_val_file4, "rb"), delimiter=",", skiprows=1);
kfolds_ave = (tv1+tv2+tv3+tv4)/4.0;
kfolds_ave = numpy.around(kfolds_ave, 6);
numpy.savetxt(kfolds_score_file, kfolds_ave, delimiter=',',comments='',fmt='%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f', header="Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train")
        
getBestEpoch(kfolds_score_file, True, EXP_NAME+'_best.csv')

#TrainVal Full - Only used to get proposal images/annotations
trainval_test_file = exp_folder+'trainval_test_score.csv'
f = open(trainval_test_file, 'a+')
f.write('Epoch, mAP-Test, F-MABO-Ave-Test, FScore-Test, MABO-Test, Recall-Test, Precision-Test, mAP-Train, F-MABO-Ave-Train, FScore-Train, MABO-Train, Recall-Train, Precision-Train\n')
f.close()
if not test_only:
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python train.py voc_boat_trainval')
for epoch in epochs:
    f = open(trainval_test_file, 'a+')
    f.write(str(epoch)+',')
    f.close()
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_test voc_boat_trainval_'+str(epoch)+' '+trainval_test_file)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+trainval_test_file+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_trainval voc_boat_trainval_'+str(epoch)+' '+trainval_test_file)
    os.system('matlab -nojvm -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';output_file=\''+trainval_test_file+'\';score;exit;catch;exit(1);end"')
    os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
    f = open(trainval_test_file, 'a+')
    #remove extra comma
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.write('\n')
    f.close()

best_epoch = getBestEpoch(trainval_test_file)
best_model_name = 'voc_boat_trainval_'+str(best_epoch)+'.h5';
os.system('CUDA_VISIBLE_DEVICES='+DEVICE+' python test.py voc_boat_test voc_boat_trainval_'+str(best_epoch))
os.system('matlab -nodesktop -nosplash -r "try;addpath(\'../HydroTestSuite/\');method=\'YOLO\';display=\'NO\';drawProposals;exit;catch;exit(1);end"')

os.system('cp models/training/yolo_boat_models/'+best_model_name + ' '+exp_folder+best_model_name)
os.system('mkdir '+ exp_folder+'proposals/')
os.system('cp -R ../HydroTestSuite/proposals/yolo/* ' + exp_folder+'proposals/')

if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')

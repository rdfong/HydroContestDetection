#Example: time python runExperiment.py 1 layerFixedExp YES NO
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

clean_models = 1;
if (CLEAN_MODELS == 'NO'):
    clean = 0;

test_only = 0;
if (TEST_ONLY == 'YES'):
    test_only = 1;

exp_folder = '../results/yolo/'+EXP_NAME+'/'

#Helper function to return the best scores from a scores list
def getBestEpoch(scoreFile, output=False, outputFile=''):
    scores = numpy.loadtxt(open(scoreFile, "rb"), delimiter=",", skiprows=1);
    #Choose based on test mAP
    best_epoch_index = numpy.argmin(scores[:,1])
    if (output == True and outFile ~= ''):
        numpy.savetxt(outputFile, scores[best_epoch_index,:], delimiter=',', header="Epoch, Test, Train, FScore, MABO, Recall, Precision")
    best_epoch = (best_epoch_index+1)*10;
    return best_epoch;

#all  output to ../results/yolo/exp_name
    #score files: 4x(kfolds), 1x (kfolds average), 1x (trainval_test)
    #best crossval average epoch stats (1 line file)
    #best trainval h5 model (potential best model)
    #image outputs and proposals on all test images (just for qualitative comparison)

epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

#Train/Val 1
train_val_file1 = exp_folder+'train1_val1_score.csv'
f = open(train_val_file1, 'a+')
f.write('Epoch, Test, Train, FScore, MABO, Recall, Precision\n')
fclose(train_val_file1)
if ~test_only:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python train.py voc_boat_train1')
for epoch in epochs:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_val1 voc_boat_train1_'+epoch+' '+train_val_file1)
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_train1 voc_boat_train1_'+epoch+' '+train_val_file1')
    os.system('matlab -nojvm < "method=\'YOLO\';output_file='+train_val_file1+';score.m"')
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')

#Train/Val 2
train_val_file2 = exp_folder+'train2_val2_score.csv'
f = open(train_val_file2, 'a+')
f.write('Epoch, Test, Train, FScore, MABO, Recall, Precision\n')
fclose(train_val_file2)
if ~test_only:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python train.py voc_boat_train2
for epoch in epochs:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_val2 voc_boat_train2_'+epoch+' '+train_val_file2)
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_train2 voc_boat_train2_'+epoch+' '+train_val_file2)
    os.system('matlab -nojvm < "method=\'YOLO\';output_file='+train_val_file2+';score.m"')
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
          
#Train/Val 3
train_val_file3 = exp_folder+'train3_val3_score.csv'
f = open(train_val_file3, 'a+')
f.write('Epoch, Test, Train, FScore, MABO, Recall, Precision\n')
fclose(train_val_file3)
if ~test_only:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python train.py voc_boat_train3
for epoch in epochs:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_val3 voc_boat_train3_'+epoch+' '+train_val_file3)
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_train3 voc_boat_train3_'+epoch+' '+train_val_file3)
    os.system('matlab -nojvm < "method=\'YOLO\';output_file='+train_val_file+';score.m"')
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
          
#Train/Val 4
train_val_file4 = exp_folder+'train4_val4_score.csv'
f = open(train_val_file4, 'a+')
f.write('Epoch, Test, Train, FScore, MABO, Recall, Precision\n')
fclose(train_val_file4)
if ~test_only:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python train.py voc_boat_train4
for epoch in epochs:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_val4 voc_boat_train4_'+epoch+' '+train_val_file4)
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_train4 voc_boat_train4_'+epoch+' '+train_val_file4)
    os.system('matlab -nojvm < "method=\'YOLO\';output_file='+train_val_file+';score.m"')
if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')

#Average KFolds results into 1 file
kfolds_score_file = exp_folder+'kfolds_average_score.csv'
tv1 = numpy.loadtxt(open(train_val_file1, "rb"), delimiter=",", skiprows=1);
tv2 = numpy.loadtxt(open(train_val_file2, "rb"), delimiter=",", skiprows=1);
tv3 = numpy.loadtxt(open(train_val_file3, "rb"), delimiter=",", skiprows=1);
tv4 = numpy.loadtxt(open(train_val_file4, "rb"), delimiter=",", skiprows=1);
kfolds_ave = (tv1+tv2+tv3+tv4)/4.0;
numpy.savetxt(kfolds_score_file, kfolds_ave, delimiter=',', header="Epoch, Test, Train, FScore, MABO, Recall, Precision")
          
getBestEpoch(kfolds_score_file, output=True, EXP_NAME+'_best.csv')

#TrainVal Full - Only used to get proposal images/annotations
trainval_test_file = exp_folder+'trainval_test_score.csv'
f = open(trainval_test_file, 'a+')
f.write('Epoch, Test, Train, FScore, MABO, Recall, Precision\n')
fclose(trainval_test_file)
if ~test_only:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python train.py voc_boat_trainval
for epoch in epochs:
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_test voc_boat_trainval_'+epoch+' '+trainval_test_file)
    os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_trainval voc_boat_trainval_'+epoch+' '+trainval_test_file)
    os.system('matlab -nojvm < "method=\'YOLO\';output_file='+trainval_test_file+';score.m"')

best_epoch = getBestEpoch(trainval_test_file)
best_model_name = 'voc_boat_trainval_'+str(best_epoch)+'.h5';
#Clear proposals folder for final run
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')
os.system('CUDA_VISIBLE_DEVICES=DEVICE python test.py voc_boat_test voc_boat_trainval_'+best_epoch)
os.system('matlab -nojvm < "method=\'YOLO\';display=\'NO\';drawProposals.m"')

os.system('cp models/training/yolo_boat_models/'+best_model_name + ' '+exp_folder+best_model_name)
os.system('cp -R ../HydroTestSuite/proposals/yolo/* ' + exp_folder+'proposals/')

if clean_models:
    os.system('rm -rf models/training/yolo_boat_models/*')
os.system('rm -rf ../HydroTestSuite/proposals/yolo/*')

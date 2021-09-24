usage: create_data_csv.py: (create csv of train/val/test)
	@arguments:
  		-h, --help            		show this help message and exit
  		--source SOURCE       		source folder of images directory
  		--random_state RANDOM_STATE	random state for train, val, test split
  		--test_size TEST_SIZE		val/test size for split
	
	@output: creates train_images.csv, validation_images.csv, test_images.csv




usage: data_generator.py: (generate data, preprocess, augment)
	@arguments:
  		-h, --help            			show this help message and exit
  		--filtname FILTNAME   			preprocessing filter name
  		--aug                 			add augmentation
  		--no_aug              			dont add augmentation
  -		-type_patch TYPE_PATCH			type of patch extraction --> by zero padding or not. By default 'pad', other opt = no_pad
  		--thick_mask_lines THICK_MASK_LINES	make mask lines thicker by number/iterations
  		--kernel_size KERNEL_SIZE		kernel size for line thickening/dilation
		--num_augmentations NUM_AUGMENTATIONS   number of augmentations
		
	@output: creates folder data_256'@filtname' with train, val, test folder of images
	Note: Augmentations are in the order of 'horizontal','rotate90','rotate180', 'rotate270', 'horizontal_rotate90', 'horizontal_rotate180', 'horizontal_rotate270'



usage: main_zones.py: (train for predicting zones + generate results)
	@arguments:
  		-h, --help            		show this help message and exit
  		--Epochs EPOCHS       		number of training epochs (integer value > 0) per run
  		--Batch_Size BATCH_SIZE		batch size (integer value)
  		--Patch_Size PATCH_SIZE		Patch size (integer value)
  		--patience PATIENCE   		patience for earlystop
  		--LOSS LOSS           		loss name--> options: binary_crossentropy, focal_loss, dice_loss, mcc_loss1,mcc_loss2, mcc_loss3,
  		--filtname FILTNAME   		preprocessing filter name
  		--max_runs MAX_RUNS   		total max num of runs to be performed sequentially (if training is too long and cannot be run in one go (24 hours limits in cluster gpu))
  		--mcc_eta MCC_ETA     		additional params for mcc loss
  		--monitor MONITOR     		monitor metric for early stop val_loss/val_mcc
		--input_path INPUT_PATH		input path name of the preprocessed images

	@output: creates quantitative results + predictions in "test" folder



usage: main_lines.py: (train for predicting lines + generate results)
	@arguments:
  		-h, --help            		show this help message and exit
  		--Epochs EPOCHS       		number of training epochs (integer value > 0) per run
  		--Batch_Size BATCH_SIZE		batch size (integer value)
  		--Patch_Size PATCH_SIZE		Patch size (integer value)
  		--patience PATIENCE   		patience for earlystop
  		--LOSS LOSS           		loss name--> options: binary_crossentropy, focal_loss, dice_loss, mcc_loss1, mcc_loss2, mcc_loss3,weighted_bce,d_map_mcc,d_map_bce,d_map_weighted_bce
  		--filtname FILTNAME   		preprocessing filter name
  		--max_runs MAX_RUNS   		total max num of runs to be performed sequentially (if training is too long and cannot be run in one go)
  		--mcc_eta MCC_ETA     		additional params for mcc loss
  		--monitor MONITOR     		monitor metric for early stop
  		--path_args PATH_ARGS		secondary path argument
  		--relax RELAX         		relaxation of d map losses
  		--class_weights CLASS_WEIGHTS	class weights for weighted bce
		--input_path INPUT_PATH		input path name of the preprocessed images
		--dilation DILATION		amount of dilation of the ground truth lines in the input path
	
	@output: creates quantitative results + predictions in path --> masks_lines_predicted_'@LOSS'_'@monitor'_R'@relax'


NOTE: The distance map loss has 3 parameters, w,R,k. While running main_lines.py you can only specify R (relax). For the other two parameters, please hardcode.




usage: Post_process.py: (Post process predicted zones)
	@arguments:
  		-h, --help            		show this help message and exit
  		--source_dir SOURCE_DIR		primary source directory based on filter/augmentation
  		--pred_dir PRED_DIR   		secondary directory i.e. predicted zones directory
	
	@output: creates postprocessed zones in folder Post_Processed inside @pred_dir



usage: Relaxed_Evaluation.py: (calculate quantitative results for a list of tolerance levels)
	@arguments:
  		-h, --help            		show this help message and exit
  		--source_dir SOURCE_DIR		primary source directory based on filter/augmentation
  -		-pred_dir PRED_DIR   		secondary directory i.e. predicted lines directory
	
	@output: creates RelaxedMetrics.txt inside @pred_dir
	NOTE: hardcode tolerance list


EXTRAS:

usage: count_imbalance.py: (counts class imbalance of source folder (train folder))
	@arguments:
  		-h, --help       	show this help message and exit
  		--source SOURCE  	source folder for counting imbalance
	
	@output: prints class imbalance



usage: create_distance_maps.py: (create distance maps for visualization/dmap algorithm)
	@arguments:
  		-h, --help            		show this help message and exit
  		--work_dir WORK_DIR   		source directory
  		--image_name IMAGE_NAME		image to be tested on
  		--w W                 		parameter w
  		--R R                 		parameter R
  		--k K                 		parameter k

	@output: creates distance map images and saves in folder distance_maps_lines


An example pipeline would be:

python3 create_data_csv.py --source JHN --random_state 1 --test_size 25
python3 data_generator.py --filtname median --aug --type_patch pad --thick_mask_lines 1 --kernel_size 5
python3 main_zones.py --Batch_Size 20 --Epochs 70 --filtname median --path_args linesw5i1 --monitor val_mcc --LOSS binary_crossentropy --max_runs 5 --patience 25
python3 main_lines.py --Batch_Size 15 --Epochs 70 --filtname median --path_args linesw5i1 --monitor val_mcc --LOSS d_map_bce --relax 2 --max_runs 5 --patience 35
python3 Post_process.py --source_dir data_256median_linesw5i1/test ---pred_dir masks_predicted_binary_crossentropy_monitor_mcc
python Relaxed_Evaluation.py --source_dir data_256median_linesw5i1/test ---pred_dir masks_lines_predicted_d_map_bce_monitor_val_mcc_R1








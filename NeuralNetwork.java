import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvANN_MLP;
import org.opencv.ml.CvANN_MLP_TrainParams;

public class NeuralNetwork {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		int num_keys = 100;
		MatOfKeyPoint keypoints = new MatOfKeyPoint();
		FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		Mat descriptors = new Mat();
		MatOfKeyPoint kp = new MatOfKeyPoint();
		Mat newdescriptors = new Mat();
		Mat temp;
		Mat image;
		Mat extra = new Mat( 1, 1, CvType.CV_32FC(7) );//CV_32FC1
		Mat in = new Mat( 1, 1, CvType.CV_8UC1 );
		int row = 0, col = 0;
		extra.put(row ,col, 0.0F,0.0F,0.0F,0.0F,0.0F,0.0F,-1.0F);
		in.put(row, col, 0);
		byte[] serial = new byte[num_keys*32];
		Mat training_data = new Mat(180,num_keys*32,CvType.CV_8UC1);
		Mat training_data_final = new Mat(num_keys*32,180,CvType.CV_8UC1);
		
		Mat test_data = new Mat(3,num_keys*32,CvType.CV_8UC1);
	//	Mat test_data_changed = new Mat(63,3,CvType.CV_32FC1);
		Mat test_data_changed = new Mat(3,num_keys*32,CvType.CV_32FC1);
		
		Mat response = new Mat(3,3,CvType.CV_8UC1);
	//	Mat response_changed = new Mat(63,3,CvType.CV_32FC1);
		Mat response_changed = new Mat(3,3,CvType.CV_32FC1);
		
	//	Mat training_data_changed = new Mat(1600,63,CvType.CV_32FC1);
		Mat training_data_changed = new Mat(180,num_keys*32,CvType.CV_32FC1);
		
		Mat output = new Mat(3,180,CvType.CV_8UC1);
		Mat output_final = new Mat(180,3,CvType.CV_8UC1);
		Mat output_changed = new Mat(180,3,CvType.CV_32FC1);
		
		
		for(int p=0;p<60;p++)
			output.put(0, p, 1);
		for(int p=60;p<180;p++)
			output.put(0, p, -1);
		
		for(int p=0;p<60;p++)
			output.put(1, p, -1);
		for(int p=60;p<120;p++)
			output.put(1, p, 1);
		for(int p=120;p<1800;p++)
			output.put(1, p, -1);
		
		for(int p=0;p<120;p++)
			output.put(2, p, -1);
		for(int p=120;p<180;p++)
			output.put(2, p, 1);
		
////////////////////rock starts//////////////////////////////////////		
		System.out.println("rock starts");		
		for(int i=1;i<=60;i++){
			temp = Highgui.imread("D:\\train_images\\rock"+i+".jpg",2);
			//Imgproc.Canny(image, temp, 10, 100, 3, true);
			if(temp.empty()){
				System.out.println("Cannot read image rock"+i);
				System.exit(0);
			}
			detector.detect(temp, keypoints);
		//	extractor.compute(temp, keypoints, descriptors);
		//	System.out.println("Size of descriptor : "+descriptors.size());
			
		/*	if(keypoints.toArray().length < 200){
				int x = 200 - keypoints.toArray().length;
				int y = keypoints.toArray().length;
				for(int i1=0;i1<x;i1++)
					keypoints.push_back(extra);
			}*/
		//	System.out.println("Length of Keypoint : "+keypoints.toArray().length);
			List<KeyPoint> listOfKeypoints = keypoints.toList();
			Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
			    @Override
			    public int compare(KeyPoint kp1, KeyPoint kp2) {
			        // Sort them in descending order, so the best response KPs will come first
			        return (int) (kp2.response - kp1.response);
			    }
			});
			
			MatOfKeyPoint keyps = new MatOfKeyPoint();
			keyps.fromList(listOfKeypoints);
			if(keyps.toArray().length < num_keys){
				int x = num_keys - keyps.toArray().length;
				int y = keyps.toArray().length;
				for(int i1=0;i1<x;i1++)
					keyps.push_back(extra);
			}
			List<KeyPoint> listOfKeyps = keyps.toList();
			
			List<KeyPoint> listOfBestKeypoints = listOfKeyps.subList(0, num_keys);
			kp.fromList(listOfBestKeypoints);
		//	System.out.println("Length of new Keypoint : "+kp.toArray().length);
			extractor.compute(temp, kp, newdescriptors);
		//	System.out.println("depth of descriptor : "+newdescriptors.depth());
		//	System.out.println("Size of descriptor : "+newdescriptors.size());
		//	System.out.println("length of serial : "+serial.length);
			newdescriptors.get(0, 0, serial);
			training_data.put(i-1, 0, serial);
		}
		
////////////////////////////////rock ends//////////////////////////////////		
		System.out.println("rock ends");		
		
////////////////////paper starts//////////////////////////////////////		
		System.out.println("paper starts");
		
for(int i=1;i<=60;i++){
temp = Highgui.imread("D:\\train_images\\ppr"+i+".jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
	System.out.println("Cannot read image ppr"+i);
	System.exit(0);
}
detector.detect(temp, keypoints);
//extractor.compute(temp, keypoints, descriptors);

/*if(keypoints.toArray().length < 200){
	int x = 200 - keypoints.toArray().length;
	int y = keypoints.toArray().length;
	for(int i1=0;i1<x;i1++)
		keypoints.push_back(extra);
}*/

List<KeyPoint> listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
    @Override
    public int compare(KeyPoint kp1, KeyPoint kp2) {
        // Sort them in descending order, so the best response KPs will come first
        return (int) (kp2.response - kp1.response);
    }
});

MatOfKeyPoint keyps = new MatOfKeyPoint();
keyps.fromList(listOfKeypoints);
if(keyps.toArray().length < num_keys){
	int x = num_keys - keyps.toArray().length;
	int y = keyps.toArray().length;
	for(int i1=0;i1<x;i1++)
		keyps.push_back(extra);
}
List<KeyPoint> listOfKeyps = keyps.toList();

List<KeyPoint> listOfBestKeypoints = listOfKeyps.subList(0, num_keys);
kp.fromList(listOfBestKeypoints);

extractor.compute(temp, kp, newdescriptors);

newdescriptors.get(0, 0, serial);
training_data.put(i+59, 0, serial);
}

////////////////////////////////paper ends//////////////////////////////////
System.out.println("paper ends");

//////////////////////////scissor starts//////////////////////////////////////		
System.out.println("scissor starts");


for(int i=1;i<=60;i++){
temp = Highgui.imread("D:\\train_images\\scr"+i+".jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image scr"+i);
System.exit(0);
}
detector.detect(temp, keypoints);
//extractor.compute(temp, keypoints, descriptors);

/*if(keypoints.toArray().length < 200){
int x = 200 - keypoints.toArray().length;
int y = keypoints.toArray().length;
for(int i1=0;i1<x;i1++)
keypoints.push_back(extra);
}*/

List<KeyPoint> listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
@Override
public int compare(KeyPoint kp1, KeyPoint kp2) {
// Sort them in descending order, so the best response KPs will come first
return (int) (kp2.response - kp1.response);
}
});

MatOfKeyPoint keyps = new MatOfKeyPoint();
keyps.fromList(listOfKeypoints);
if(keyps.toArray().length < num_keys){
	int x = num_keys - keyps.toArray().length;
	int y = keyps.toArray().length;
	for(int i1=0;i1<x;i1++)
		keyps.push_back(extra);
}
List<KeyPoint> listOfKeyps = keyps.toList();

List<KeyPoint> listOfBestKeypoints = listOfKeyps.subList(0, num_keys);
kp.fromList(listOfBestKeypoints);

extractor.compute(temp, kp, newdescriptors);

newdescriptors.get(0, 0, serial);
training_data.put(i+119, 0, serial);
}

////////////////////////////////scissor ends//////////////////////////////////
System.out.println("scissor ends");

///////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////// test data starts//////////////////////////////////////		
System.out.println(" test data starts");


temp = Highgui.imread("D:\\images\\r23.jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image r23");
System.exit(0);
}
detector.detect(temp, keypoints);
//extractor.compute(temp, keypoints, descriptors);

/*if(keypoints.toArray().length < 200){
int x = 200 - keypoints.toArray().length;
int y = keypoints.toArray().length;
for(int i1=0;i1<x;i1++)
keypoints.push_back(extra);
}*/

List<KeyPoint> listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
@Override
public int compare(KeyPoint kp1, KeyPoint kp2) {
// Sort them in descending order, so the best response KPs will come first
return (int) (kp2.response - kp1.response);
}
});

MatOfKeyPoint keyps1 = new MatOfKeyPoint();
keyps1.fromList(listOfKeypoints);
if(keyps1.toArray().length < num_keys){
	int x = num_keys - keyps1.toArray().length;
	int y = keyps1.toArray().length;
	for(int i1=0;i1<x;i1++)
		keyps1.push_back(extra);
}
List<KeyPoint> listOfKeyps1 = keyps1.toList();

List<KeyPoint> listOfBestKeypoints = listOfKeyps1.subList(0, num_keys);
kp.fromList(listOfBestKeypoints);

extractor.compute(temp, kp, newdescriptors);

newdescriptors.get(0, 0, serial);
test_data.put(0, 0, serial);

///----------------------------------------------------------------------//

temp = Highgui.imread("D:\\images\\p22.jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image p23");
System.exit(0);
}
detector.detect(temp, keypoints);
//extractor.compute(temp, keypoints, descriptors);

/*if(keypoints.toArray().length < 200){
int x = 200 - keypoints.toArray().length;
int y = keypoints.toArray().length;
for(int i1=0;i1<x;i1++)
keypoints.push_back(extra);
}*/

listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
@Override
public int compare(KeyPoint kp1, KeyPoint kp2) {
// Sort them in descending order, so the best response KPs will come first
return (int) (kp2.response - kp1.response);
}
});

MatOfKeyPoint keyps2 = new MatOfKeyPoint();
keyps2.fromList(listOfKeypoints);
if(keyps2.toArray().length < num_keys){
	int x = num_keys - keyps2.toArray().length;
	int y = keyps2.toArray().length;
	for(int i1=0;i1<x;i1++)
		keyps2.push_back(extra);
}
List<KeyPoint> listOfKeyps2 = keyps2.toList();

listOfBestKeypoints = listOfKeyps2.subList(0, num_keys);
kp.fromList(listOfBestKeypoints);

extractor.compute(temp, kp, newdescriptors);

newdescriptors.get(0, 0, serial);
test_data.put(1, 0, serial);

//--------------------------------------------------------------------------//


temp = Highgui.imread("D:\\images\\s24.jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image s24");
System.exit(0);
}
detector.detect(temp, keypoints);
//extractor.compute(temp, keypoints, descriptors);

/*if(keypoints.toArray().length < 200){
int x = 200 - keypoints.toArray().length;
int y = keypoints.toArray().length;
for(int i1=0;i1<x;i1++)
keypoints.push_back(extra);
}*/

listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
@Override
public int compare(KeyPoint kp1, KeyPoint kp2) {
// Sort them in descending order, so the best response KPs will come first
return (int) (kp2.response - kp1.response);
}
});

MatOfKeyPoint keyps3 = new MatOfKeyPoint();
keyps3.fromList(listOfKeypoints);
if(keyps3.toArray().length < num_keys){
	int x = num_keys - keyps3.toArray().length;
	int y = keyps3.toArray().length;
	for(int i1=0;i1<x;i1++)
		keyps3.push_back(extra);
}
List<KeyPoint> listOfKeyps3 = keyps3.toList();

listOfBestKeypoints = listOfKeyps3.subList(0, num_keys);
kp.fromList(listOfBestKeypoints);

extractor.compute(temp, kp, newdescriptors);

newdescriptors.get(0, 0, serial);
test_data.put(2, 0, serial);


////////////////////////////////test data ends//////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

Core.transpose(training_data, training_data_final);
Core.transpose(output, output_final);

		
		Mat layerSizes = new Mat(3,1,CvType.CV_32S); //Setting up the layers
		layerSizes.put(0,0,num_keys*32);
		layerSizes.put(1, 0, 30);
		layerSizes.put(2, 0, 3);
		
		CvANN_MLP nnet2 = new CvANN_MLP(layerSizes,CvANN_MLP.SIGMOID_SYM,0.7,1);
		CvANN_MLP_TrainParams params = new CvANN_MLP_TrainParams();
		TermCriteria criteria = new TermCriteria(TermCriteria.COUNT+TermCriteria.EPS,1000,0.000001);
		params.set_train_method(CvANN_MLP_TrainParams.BACKPROP);
		params.set_bp_dw_scale(0.1);
		params.set_bp_moment_scale(0.1);
		params.set_term_crit(criteria);  
		training_data.convertTo(training_data_changed, CvType.CV_32FC1);
		output_final.convertTo(output_changed, CvType.CV_32FC1);
		int iter = nnet2.train(training_data_changed, output_changed,new Mat(), new Mat() ,params, 0);
		System.out.println("Number of iteratins: "+ iter);
		nnet2.save("D:\\nnet_22.xml");
	/*	try
	      {
	         FileOutputStream fileOut = new FileOutputStream("D:\\nnet.net");
	         ObjectOutputStream out = new ObjectOutputStream(fileOut);
	         
	         out.writeObject(nnet);
	         out.close();
	         fileOut.close();
	         System.out.printf("neural network saved");
	      }catch(IOException i)
	      {
	          i.printStackTrace();
	      }*/
		System.out.println("Running test...");
		test_data.convertTo(test_data_changed, CvType.CV_32FC1);
		response.convertTo(response_changed, CvType.CV_32FC1);
		nnet2.predict(test_data_changed,response_changed);
		System.out.println(response_changed.dump());

System.out.println("Size of training data : "+training_data.size());
System.out.println("Size of training data : "+training_data_final.size());

	}

}

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvANN_MLP;

public class TestNeuralNetwork {

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
		Mat temp = new Mat();
		Mat image;
		Mat extra = new Mat( 1, 1, CvType.CV_32FC(7) );//CV_32FC1
		int row = 0, col = 0;
		extra.put(row ,col, 0.0F,0.0F,0.0F,0.0F,0.0F,0.0F,-1.0F);
		byte[] serial = new byte[num_keys*32];
		Mat test_data = new Mat(25,serial.length,CvType.CV_8UC1);
		Mat test_data_changed = new Mat(25,serial.length,CvType.CV_32FC1);
		Mat response = new Mat(25,3,CvType.CV_8UC1);
		Mat response_changed = new Mat(25,3,CvType.CV_32FC1);
		
		
////////////////////rock starts//////////////////////////////////////		
System.out.println("rock starts");		
for(int i=61;i<=67;i++){
temp = Highgui.imread("D:\\train_images\\rock"+i+".jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
	System.out.println("Cannot read image rock"+i);
	System.exit(0);
}
detector.detect(temp, keypoints);

List<KeyPoint> listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
    @Override
    public int compare(KeyPoint kp1, KeyPoint kp2) {
        
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
test_data.put(i-61, 0, serial);
}

////////////////////////////////rock ends//////////////////////////////////		
System.out.println("rock ends");


////////////////////paper starts//////////////////////////////////////		
System.out.println("paper starts");

for(int i=71;i<=77;i++){
temp = Highgui.imread("D:\\train_images\\ppr"+i+".jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image ppr"+i);
System.exit(0);
}
detector.detect(temp, keypoints);

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
test_data.put(i-71+7, 0, serial);
}

////////////////////////////////paper ends//////////////////////////////////
System.out.println("paper ends");



//////////////////////////scissor starts//////////////////////////////////////		
System.out.println("scissor starts");


for(int i=81;i<=91;i++){
temp = Highgui.imread("D:\\train_images\\scr"+i+".jpg",2);
//Imgproc.Canny(image, temp, 10, 100, 3, true);
if(temp.empty()){
System.out.println("Cannot read image scr"+i);
System.exit(0);
}
detector.detect(temp, keypoints);

List<KeyPoint> listOfKeypoints = keypoints.toList();
Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
@Override
public int compare(KeyPoint kp1, KeyPoint kp2) {
//Sort them in descending order, so the best response KPs will come first
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
test_data.put(i-81+14, 0, serial);
}

////////////////////////////////scissor ends//////////////////////////////////
System.out.println("scissor ends");


CvANN_MLP nnet = new CvANN_MLP();
nnet.load("D:\\nnet_22.xml");

test_data.convertTo(test_data_changed, CvType.CV_32FC1);
response.convertTo(response_changed, CvType.CV_32FC1);

System.out.println("Running test...");
nnet.predict(test_data_changed,response_changed);
System.out.println(response_changed.dump());

/////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/////////////
	}

}

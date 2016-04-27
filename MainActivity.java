package com.example.murtazachunia.cameratest;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.*;
import android.os.Process;
import android.provider.Settings;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.ml.CvANN_MLP;

import java.io.File;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


public class MainActivity extends AppCompatActivity implements Runnable{

   // BaseLoaderCallback mLoaderCallback;
    Button play;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        play = (Button)findViewById(R.id.btnplay);
        play.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Thread thread = new Thread(MainActivity.this);
                thread.setDaemon(true);
                thread.start();
            }
        });

    /*    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        VideoCapture camera = new VideoCapture(0);

        if(!camera.isOpened()){
            Toast.makeText(MainActivity.this,"Camera cant be opened",Toast.LENGTH_LONG).show();
            System.exit(0);
        }
        else{
            Mat frame = new Mat();
            while(true){
                if(camera.read(frame)){
                    Imgcodecs.imwrite("/sdcard/Pictures/Screenshots/opencv.jpg",frame);
                    Toast.makeText(MainActivity.this,"Image written....",Toast.LENGTH_LONG).show();
                    break;
                }
            }
        }*/
    }

    public void onResume(){
        super.onResume();

    }

    public void onStop(){
        super.onStop();
       // Intent intent = new Intent(MainActivity.this,MyService.class);
        //stopService(intent);
    }

    @Override
    public void run() {
        android.os.Process.setThreadPriority(Process.THREAD_PRIORITY_FOREGROUND);
       // Intent intent = new Intent(MainActivity.this,MyService.class);
       // startService(intent);

        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_11, this, mLoaderCallback);



    }


    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                    //  Log.i("opencv", "OpenCV loaded successfully");
                    //  mOpenCvCameraView.enableView();
                    // mOpenCvCameraView.setOnTouchListener( MainActivity.this );
                    VideoCapture camera = new VideoCapture(Highgui.CV_CAP_ANDROID+1);
                    final Context context = getApplicationContext();
                    if(!camera.isOpened()){

                        Handler handler = new Handler();
                        handler.post(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(MainActivity.this, "Camera cant be opened", Toast.LENGTH_LONG).show();
                            }
                        });
                    }
                    else{
                        Handler handler = new Handler();
                        handler.post(new Runnable() {
                            @Override
                            public void run() {
                             //   Toast.makeText(MainActivity.this, "in thread.....", Toast.LENGTH_LONG).show();

                            }
                        });

                        final   Mat frame = new Mat();
                       // camera.set(Highgui.CV_CAP_PROP_FRAME_WIDTH, 30);
                       // camera.set(Highgui.CV_CAP_PROP_FRAME_HEIGHT, 40);
                       /* while(true){
                            if(camera.read(frame)){
                                Highgui.imwrite("/sdcard/Pictures/Screenshots/opencv.jpg", frame);
                                Toast.makeText(context,"Image written....",Toast.LENGTH_LONG).show();
                                break;
                            }
                        }*/
                        boolean grabbed = camera.grab();
                        if (grabbed) {
                            //  Bitmap image = null;
                            camera.retrieve(frame, Highgui.CV_CAP_ANDROID_GREY_FRAME);
                            int num_keys = 160;
                            Mat extra = new Mat( 1, 1, CvType.CV_32FC(7) );//CV_32FC1
                            int row = 0, col = 0;
                            extra.put(row, col, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -1.0F);
                            byte[] serial = new byte[num_keys*32];
                            Mat test_data = new Mat(1,serial.length,CvType.CV_8UC1);
                            Mat test_data_changed = new Mat(1,serial.length,CvType.CV_32FC1);
                            Mat response = new Mat(1,3,CvType.CV_8UC1);
                            Mat response_changed = new Mat(1,3,CvType.CV_32FC1);


                            MatOfKeyPoint keypoints = new MatOfKeyPoint();
                            FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
                            detector.detect(frame, keypoints);

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
                            keypoints.fromList(listOfBestKeypoints);

                            DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                            final Mat descriptors = new Mat();
                            extractor.compute(frame, keypoints, descriptors);
                            descriptors.get(0, 0, serial);
                            test_data.put(0, 0, serial);
                            test_data.convertTo(test_data_changed, CvType.CV_32FC1);
                            response.convertTo(response_changed, CvType.CV_32FC1);
                            double yn = Math.random();
                            float rn = (float)yn;
                            String computer = new String();
                            if(rn > 0.33 && rn < 0.67){
                                computer = "PAPER";
                            }
                            else if(rn >= 0.67){
                                computer = "SCISSOR";
                            }
                            else computer = "ROCK";
                            //   Utils.bitmapToMat(image, frame);
                            //CV_CAP_ANDROID_COLOR_FRAME_RGB
                           // Highgui.imwrite("/sdcard/Pictures/Screenshots/opencv.jpg", frame);
                          /*  FileOutputStream out = null;
                            try {
                                File file = new File("/sdcard/Pictures/Screenshots/image.jpg");
                                out = new FileOutputStream(file);
                                image.compress(Bitmap.CompressFormat.JPEG, 85, out); // bmp is your Bitmap instance
                                // PNG is a lossless format, the compression factor (100) is ignored
                            } catch (Exception e) {
                                e.printStackTrace();
                            } finally {
                                try {
                                    if (out != null) {
                                        out.close();
                                    }
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }*/

                          //  Uri path = Uri.parse("android.resource://com.example.murtazachunia.cameratest/" + R.raw.nnet2);
                            //String uriPath = "android.resource://"+getPackageName()+"/raw/nnet2";
                            CvANN_MLP nnet = new CvANN_MLP();

                           // final String mesg = path.toString();
                            final String mesg2 = context.getResources().toString();
                            File netdir = context.getDir("raw", Context.MODE_PRIVATE);
                           // File netfile = new File(netdir, "nnet_22.xml");
                            //final String mesg3 = netfile.toString();

                            nnet.load("/sdcard/Pictures/Screenshots/nnet_20.xml");//specify the file path on your phone here....
                            nnet.predict(test_data_changed,response_changed);
                            final float[] result = new float[3];
                            //final String mesg4 = response_changed.dump().toString();
                           // final String mesg4 = result[0];
                            final String mesg_final;
                            response_changed.get(0,0,result);
                            int x = Float.compare(result[0], result[1]);
                            if(x>0){
                                x = Float.compare(result[0],result[2]);
                                if(x > 0) {

                                    mesg_final = "ROCK";
                                }
                                else mesg_final = "SCISSOR";
                            }
                            else {
                                x = Float.compare(result[1],result[2]);
                                if(x > 0){
                                    mesg_final = "PAPER";
                                }
                                else mesg_final = "SCISSOR";
                            }
                            final String finale;
                            if(computer.equals("ROCK") && mesg_final.equals("SCISSOR")){
                                finale = "Computer predicted rock..computer wins";
                            }
                            else if(computer.equals("ROCK") && mesg_final.equals("PAPER")){
                                finale = "Computer predicted rock..You wins";
                            }
                            else if(computer.equals("ROCK") && mesg_final.equals("ROCK")){
                                finale = "Computer predicted rock..Its a TIE";
                            }
                            else if(computer.equals("PAPER") && mesg_final.equals("SCISSOR")){
                                finale = "Computer predicted paper..You wins";
                            }
                            else if(computer.equals("PAPER") && mesg_final.equals("PAPER")){
                                finale = "Computer predicted paper..Its a TIE";
                            }
                            else if(computer.equals("PAPER") && mesg_final.equals("ROCK")){
                                finale = "Computer predicted paper..computer wins";
                            }
                            else if(computer.equals("SCISSOR") && mesg_final.equals("SCISSOR")){
                                finale = "Computer predicted scissor..Its a TIE";
                            }
                            else {
                                if(computer.equals("SCISSOR") && mesg_final.equals("PAPER"))
                                finale = "Computer predicted scissor..computer wins";
                                else finale = "Computer predicted scissor..You win";
                            }






                            Handler handler2 = new Handler();
                            handler2.post(new Runnable() {
                                @Override
                                public void run() {
                                   // Toast.makeText(MainActivity.this, mesg_final+" "+result[0]+" "+result[1]+" "+result[2], Toast.LENGTH_LONG).show();
                                    Toast.makeText(MainActivity.this, finale, Toast.LENGTH_LONG).show();
                                            //depth: " + descriptors.depth() + " channel: " + frame.channels()
                                            //"Image written.."+frame.size()+" depth: " + descriptors.depth() + " channel: " + frame.channels()
                                }
                            });

                            camera.release();
                        }
                    }
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }

        }
    };
}


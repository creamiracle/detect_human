#include <deep_object_detection/DetectObjects.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <iostream>
#include <tf/LinearMath/Vector3.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/distances.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "boost/date_time/posix_time/posix_time.hpp" //include all types plus i/o
#include "boost/date_time/posix_time/posix_time_types.hpp" //no i/o just types
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Pose.h>
#include <sstream>
#include <math.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;

tf::StampedTransform transform; 

//tf::TransformListener listener;

ros::ServiceClient client;

Cloud pcl_depth_reg_data;

sensor_msgs::Image depth_image;

tf::Vector3 robot_position; //need to improve

std::ifstream infile;

visualization_msgs::MarkerArray poseMarker;

int count = 0;

tf::Vector3 robot_pose;

int human_counter = 0;

Cloud crop3DObjectFromPointCloud(const deep_object_detection::Object& object, Cloud objectcloud, int image_cols,tf::Vector3 robotPosition,tf::StampedTransform transform, int human_counter)
{
  Cloud objectpc;
  ROS_INFO("head name %s", objectcloud.header.frame_id.c_str());
  ROS_INFO("start function");
  for(int x = object.x; x < object.x + object.width; x++)
  {
    for(int y = object.y; y < object.y + object.height; y++)
      {
        PointType point = objectcloud.points[y*image_cols + x];

        if(pcl_isfinite(point.x))
        {
          Eigen::Vector4f pointxy;
          pointxy[0] = point.x;
          pointxy[1] = point.y;
          pointxy[2] = 0.0;
          pointxy[3] = 0.0;

          Eigen::Vector4f robotxy;

          robotxy[0] = robotPosition[0];
          robotxy[1] = robotPosition[1];
          robotxy[2] = 0.0;
          robotxy[3] = 0.0;

          double dist = pcl::distances::l2(pointxy,robotxy);

          if(dist <= 4.0)
            objectpc.push_back(point);
        }
      }
    }

  std::stringstream ss;
  ss<<"human_cloud"<<human_counter<<".pcd";
  pcl::io::savePCDFileASCII (ss.str(), objectpc);
  Eigen::Vector4f centroid;

  pcl::compute3DCentroid(objectpc,centroid);

  ROS_INFO("The distance of the person %f %f %f",centroid[0],centroid[1],centroid[2]);

  

  
  tf::Vector3 point(centroid[0], centroid[1], centroid[2]); 

    
  //point in base_link
  tf::Vector3 point_bl = transform * point;
  ROS_INFO("person in base_link %f %f %f",point_bl[0],point_bl[1],point_bl[2]);

  //print time into 2 format secs+nsec date
  //ros::Time thistime = objectcloud.header.stamp;
  ros::Time thistime = ros::Time::now();
  //ROS_INFO("time now is %s",nowtime.c_str());
  //ros::Time thistime = transform.stamp_;
  double secs = thistime.toSec();
  uint64_t nsecs = thistime.toNSec();
  //ROS_INFO("time sec %d %d", secs, nsecs);
  
  boost::posix_time::ptime thistimeboost = thistime.toBoost();
  std::string thistimestr = to_simple_string(thistimeboost);
  ROS_INFO("time %s",thistimestr.c_str());

  //show in a MarkerArray
  visualization_msgs::Marker aMarker;
  aMarker.header.frame_id = "/map";
  aMarker.header.stamp = transform.stamp_;
  aMarker.pose.position.x = point_bl[0];
  aMarker.pose.position.y = point_bl[1];
  aMarker.pose.position.z = point_bl[2];
  aMarker.type = aMarker.ARROW;
  aMarker.scale.x = 0.5;
  aMarker.scale.y = 0.1;
  aMarker.scale.z = 0.1;
  aMarker.color.a = 1;
  aMarker.color.g = 1;
  aMarker.id = count;
  aMarker.ns = "sector";

  count++;
  poseMarker.markers.push_back(aMarker);

  //save in txt
  std::ofstream outfile("/home/lin/catkin_ws/src/detect_human/result/resultbag2.txt", std::ios_base::app);
  if(!outfile)
  {
    std::cout<<"error";
  }
  else
  {
    outfile << point_bl[0] << "," << point_bl[1] << "," << point_bl[2] << " " << secs << "," << nsecs << "," << thistimestr <<endl;
    outfile.close();
  }

  return objectpc;
}

// robot position call back
void robotPoseCallback(const geometry_msgs::Pose::ConstPtr& pose)
{
  //ROS_INFO("The distance of the person in base_link %f %f %f",point_bl[0],point_bl[1],point_bl[2]);
  tf::Vector3 point(pose->position.x, pose->position.y, pose->position.z); 
  robot_position = transform * point;
  //ROS_INFO("The new robot position %f %f %f",point_bl[0],point_bl[1],point_bl[2]);
}

bool calculatePositionfromDepthImage(int x, int y, int width, int height)
{
  bool flag = true;
  cv_bridge::CvImagePtr cv_ptr;
    //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        //Always copy, returning a mutable CvImage
        //OpenCV expects color images to use BGR channel order.
        cv_ptr = cv_bridge::toCvCopy(depth_image);
    }
    catch (cv_bridge::Exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
        return false;
    }

    cv::Mat depth_float_img = cv_ptr->image;

    int bbox_c_x = x+width/2;
    int bbox_c_y = y+height/2;

    float posz = depth_float_img.at<float>(bbox_c_y,bbox_c_x);
    if(posz != posz)
    {
      posz = 0;
    }
    int counter = 1;

    //ROS_INFO("posez before is %f",posz);
    for(int i = -5 ; i < 6; i++)
    {
      //make sure not a nan
      if(i != 0)
      {
        if(depth_float_img.at<float>(bbox_c_y + i,bbox_c_x) == depth_float_img.at<float>(bbox_c_y + i,bbox_c_x))
        {
          //ROS_INFO("adding %f",depth_float_img.at<float>(bbox_c_y + i,bbox_c_x));
          posz += depth_float_img.at<float>(bbox_c_y + i,bbox_c_x);
          counter++;
        }
      }
    }
    for(int j = -5 ; j < 6; j++)
    {
      if(j != 0)
      {
        if(depth_float_img.at<float>(bbox_c_y,bbox_c_x + j) == depth_float_img.at<float>(bbox_c_y,bbox_c_x + j))
        {
          //ROS_INFO("adding %f",depth_float_img.at<float>(bbox_c_y,bbox_c_x + j));
          posz += depth_float_img.at<float>(bbox_c_y,bbox_c_x + j);
          counter++;
        }
      }
    }  
    
    posz = posz / counter;


    //ROS_INFO("posez now is %f",posz);

    int image_center_y = depth_float_img.rows/2;
    int image_center_x = depth_float_img.cols/2;

    int diffx = bbox_c_x - image_center_x;
    int diffy = bbox_c_y - image_center_y;

    float anglexrad = atan2(diffx,525.0);
    float angleyrad = atan2(diffy,525.0);

    float posx = posz*tan(anglexrad);
    float posy = posz*tan(angleyrad);

    if(posz != posz  || posx != posx || posy != posy)
    {
      ROS_WARN("NaN value observed");
      //ROS_INFO("anglex is %f, angley is %f",anglexrad, angleyrad);
      //ROS_INFO("posx is %f, posy is %f. posz is %f", posx, posy, posz);
      return false;
    }

    if(posz == 0)
    {
      ROS_WARN("all nan value return 0");
      return false;
    }

    if(posz > 5.0)
      return false;

    ROS_INFO("Position of human %.2f %.2f %.2f",posx,posy,posz); 

    tf::Vector3 point(posx, posy, posz); 

    
  //point in base_link
    tf::Vector3 point_bl = transform * point;
    ROS_INFO("Position in base_link %f %f %f",point_bl[0],point_bl[1],point_bl[2]);

  //print time into 2 format secs+nsec date
  //ros::Time thistime = objectcloud.header.stamp;
    ros::Time thistime = ros::Time::now();
  //ROS_INFO("time now is %s",nowtime.c_str());
  //ros::Time thistime = transform.stamp_;
    double secs = thistime.toSec();
    uint64_t nsecs = thistime.toNSec();
    //ROS_INFO("time sec %.2f %.2f", secs, nsecs);

    boost::posix_time::ptime thistimeboost = thistime.toBoost();
    std::string thistimestr = to_simple_string(thistimeboost);
    ROS_INFO("time %s",thistimestr.c_str());

  //show in a MarkerArray
    visualization_msgs::Marker aMarker;
    aMarker.header.frame_id = "/map";
    aMarker.header.stamp = transform.stamp_;
    aMarker.pose.position.x = point_bl[0];
    aMarker.pose.position.y = point_bl[1];
    aMarker.pose.position.z = point_bl[2];
    aMarker.type = aMarker.CUBE;
    aMarker.scale.x = 0.5;
    aMarker.scale.y = 0.1;
    aMarker.scale.z = 0.1;
    aMarker.color.a = 1;
    aMarker.color.g = 1;
    aMarker.id = count;
    aMarker.ns = "sector";

    count++;
    poseMarker.markers.push_back(aMarker);

  //save in txt
    //std::ofstream outfile("/home/lin/catkin_ws/src/detect_human/result/newposition.txt", std::ios_base::app);
    std::ofstream outfile("/home/lin/catkin_ws/src/detect_human/result/result20131114.txt", std::ios_base::app);
    if(!outfile)
    {
      std::cout<<"error";
    }
    else
    {
      outfile << point_bl[0] << "," << point_bl[1] << "," << point_bl[2] << " " << secs << "," << nsecs << "," << thistimestr <<endl;
      outfile.close();
    }
  return flag;
}


void depthRpcCallback(const sensor_msgs::PointCloud2::ConstPtr& depthRpc)
{
  ROS_INFO("get the depth call back");
  pcl::fromROSMsg(*depthRpc, pcl_depth_reg_data);
}

void depthToCV8UC1(const cv::Mat& float_img, cv::Mat& mono8_img)
{
  //Process images
  if(mono8_img.rows != float_img.rows || mono8_img.cols != float_img.cols){
    mono8_img = cv::Mat(float_img.size(), CV_8UC1);}
  cv::convertScaleAbs(float_img, mono8_img, 100, 0.0);
}

void depthImgCallBack(const sensor_msgs::Image::ConstPtr& msg)
{
   depth_image = *msg;

   cv_bridge::CvImagePtr cv_ptr;
    //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        //Always copy, returning a mutable CvImage
        //OpenCV expects color images to use BGR channel order.
        cv_ptr = cv_bridge::toCvCopy(depth_image);
        cv_ptr->encoding = "bgr8";
    }
    catch (cv_bridge::Exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
        return;
    }

    //Copy the image.data to imageBuf.
    cv::Mat depth_float_img = cv_ptr->image;
    cv::Mat depth_mono8_img;
    depthToCV8UC1(depth_float_img, depth_mono8_img);
   // cv::imshow("depth",depth_mono8_img);
   // cv::waitKey(10);
}

void imageCallback(const sensor_msgs::Image::ConstPtr& image)
{
  ROS_INFO("get the image call back");
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  deep_object_detection::DetectObjects srv;

  std::vector<sensor_msgs::Image> images;
  images.push_back(*image);

  srv.request.images = images;
  srv.request.confidence_threshold = 0.6;

  if(client.call(srv))
  {
    std::cout<<srv.response.objects.size()<<std::endl;
    for(int i = 0; i < srv.response.objects.size() ; i++)
    {
      ROS_INFO("find a %s",srv.response.objects[i].label.c_str());
      if(srv.response.objects[i].label == "person")
      {
        if(pcl_depth_reg_data.points.size() > 0)
        {
          bool flag;
          cv::Point pt1;
          pt1.x = srv.response.objects[i].x;
          pt1.y = srv.response.objects[i].y;
          cv::Point pt2;
          pt2.x = srv.response.objects[i].x + srv.response.objects[i].width ;
          pt2.y = srv.response.objects[i].y + srv.response.objects[i].height;
          cv::rectangle(cv_ptr->image, pt1, pt2, cv::Scalar(255,255,0), 2, 8, 0 );
          flag = calculatePositionfromDepthImage(srv.response.objects[i].x,srv.response.objects[i].y,srv.response.objects[i].width,srv.response.objects[i].height);
          //crop3DObjectFromPointCloud(srv.response.objects[i], pcl_depth_reg_data, cv_ptr->image.cols, robot_position, transform,human_counter);
          if(flag == true)
          {
            std::stringstream ss;
            ss<<"human_image"<<human_counter<<".jpg";
            cv::imwrite(ss.str().data(),cv_ptr->image);
            human_counter++;
          }
        }
      }
    }
  }

  //cv::imshow("image",cv_ptr->image);
  //cv::waitKey(10);

}

int main(int argc, char *argv[])
{
  Cloud objectpc;
  Cloud pcl_depth_data, pcl_depth_reg_data;

  ros::init(argc, argv, "detect_human");

  ros::NodeHandle n;
  ros::Rate loop_rate(10);

  robot_position[0] = 0.0;
  robot_position[1] = 0.0;
  robot_position[2] = 0.0;
  ROS_INFO("start running");

  //ros::Subscriber robot_pose_sub = n.subscribe("/robot_pose ",1, robotPoseCallback);
  //ros::Subscriber depth_rpc_sub = n.subscribe("/camera/depth/points", 1, depthRpcCallback);
  //ros::Subscriber image_sub = n.subscribe("/camera/rgb/image_rect_color", 1, imageCallback);
  ros::Subscriber depth_rpc_sub = n.subscribe("/head_xtion/depth/points", 1, depthRpcCallback);
  ros::Subscriber image_sub = n.subscribe("/head_xtion/rgb/image_color" , 1, imageCallback);
  ros::Subscriber depth_image_sub = n.subscribe("/head_xtion/depth/image", 1, depthImgCallBack);

  client = n.serviceClient<deep_object_detection::DetectObjects>("deep_object_detection/detect_objects");
  ros::Publisher pose_pub = n.advertise<visualization_msgs::MarkerArray>("/robot_position", 1);


    //transform
  tf::TransformListener listener;

 
    
  while(ros::ok())
  {
    pose_pub.publish(poseMarker);
    ros::spinOnce();
    loop_rate.sleep();


  
  try
  {
    //-----------------------------------base_link or map???---------------------------------------
    listener.waitForTransform("map", "head_xtion_depth_optical_frame", ros::Time::now(), ros::Duration(5.0));
    listener.lookupTransform("map", "head_xtion_depth_optical_frame", ros::Time(0) ,transform);
  }
  catch(tf::TransformException e)
  {
    ROS_ERROR("%s", e.what());
  }
  }

  ros::spin();

  return 0;
}

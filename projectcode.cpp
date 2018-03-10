/*
Program name    : Object recognition using Eigenface algorithm as a part of applied research project.
Author          : Job Vakayil Jose, Sarath Marson, HSRW Kleve, 47533, Germany.
Date            : March 2018
Version		    : 2.0

This program uses open library named OpenCV and its all rights goes to openCV developers

//////////////////////////////////////////////////////////////////////////////////////////
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.          			//
//  By downloading, copying, installing or using the software you agree to this license.//
//  If you do not agree to this license, do not download, install,          			//
//  copy or use the software.                           								//
//									                                            		//
//                          License Agreement                   						//
//                For Open Source Computer Vision Library               				//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.	            		//
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.         				//
// Third party copyrights are property of their respective owners.	            		//
// Redistribution and use in source and binary forms, with or without modification, 	//
// are permitted provided that the following conditions are met:            			//
//   * Redistribution's of source code must retain the above copyright notice,	    	//
//     this list of conditions and the following disclaimer.            				//
//   * Redistribution's in binary form must reproduce the above copyright notice,   	//
//     this list of conditions and the following disclaimer in the documentation	    //
//     and/or other materials provided with the distribution.			            	//
//   * The name of the copyright holders may not be used to endorse or promote products	//
// derived from this software without specific prior written permission.        		//
// This software is provided by the copyright holders and contributors "as is" and  	//
// any express or implied warranties, including, but not limited to, the implied    	//
// warranties of merchantability and fitness for a particular purpose are disclaimed.   //
// In no event shall the Intel Corporation or contributors be liable for any direct,	//
// indirect, incidental, special, exemplary, or consequential damages		        	//
// (including, but not limited to, procurement of substitute goods or services;	    	//
// loss of use, data, or profits; or business interruption) however caused	        	//
// and on any theory of liability, whether in contract, strict liability,	        	//
// or tort (including negligence or otherwise) arising in any way out of	        	//
// the use of this software, even if advised of the possibility of such damage.         //

*/



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

int main()
{

    printf(" Object recognition program using eigenface algorithm.\n ______________________________________________________ \n \n Developed by Job Vakayil Jose and Sarath Marson \n OpenCV open source computer vision library is used in this program\n OpenCV copyright belongs to its developers opencv.org \n ______________________________________________________  \n \n");
    printf(" This program compares given input image file with \n 4 image files which are given in the database file. \n ______________________________________________________  \n \n If you want to change database files, \n open path.txt and change the file paths \n ______________________________________________________ \n \n");
    int i=0,j=0,k=0;
    int i_horiz=20,i_vert=20,no_files=4;

    string textfilepath;
    textfilepath="path.txt";
    ifstream file(textfilepath.c_str()); //opens the input file for reading
    string str;
    string name[4];

	//following code is used to collect file path from given text file
    while (getline(file, str))
    {
        name[i]=str;
        i++;
    }

    Mat file01,file02,file03,file04; //Opencv class mat for each file

    //following function is used to get gray scale pixel values to corresponding class
    file01=imread(name[0],IMREAD_GRAYSCALE);
    file02=imread(name[1],IMREAD_GRAYSCALE);
    file03=imread(name[2],IMREAD_GRAYSCALE);
    file04=imread(name[3],IMREAD_GRAYSCALE);

    // creating arrays for each file (20x20)
    double file01_array[i_horiz][i_vert];
    double file02_array[i_horiz][i_vert];
    double file03_array[i_horiz][i_vert];
    double file04_array[i_horiz][i_vert];

    double file_mean[i_horiz][i_vert]; //array for mean (20x20)

    // Loops for finding the mean value of pixels and grey values

    for(i=0;i<i_horiz;i++)
        {
            for(j=0;j<i_vert;j++)
                {
                   file01_array[i][j]=file01.at<uchar>(i, j);
                   file02_array[i][j]=file02.at<uchar>(i, j);
                   file03_array[i][j]=file03.at<uchar>(i, j);
                   file04_array[i][j]=file04.at<uchar>(i, j);


                   file_mean[i][j]=(file01_array[i][j]+file02_array[i][j]+file03_array[i][j]+file04_array[i][j])/4;
                }
        }


        // loops to find out the mean free matrix(b) (400x4) and its transpose (4x400)

        k=0;
        double b[i_horiz*i_vert][no_files];
        double b_transpose[no_files][i_horiz*i_vert];

        for(i=0;i<i_horiz;i++)
            {
                for(j=0;j<i_vert;j++)
                    {
                       b[k][0]=file01_array[i][j]-file_mean[i][j];
                       b[k][1]=file02_array[i][j]-file_mean[i][j];
                       b[k][2]=file03_array[i][j]-file_mean[i][j];
                       b[k][3]=file04_array[i][j]-file_mean[i][j];


                       b_transpose[0][k]=file01_array[i][j]-file_mean[i][j];
                       b_transpose[1][k]=file02_array[i][j]-file_mean[i][j];
                       b_transpose[2][k]=file03_array[i][j]-file_mean[i][j];
                       b_transpose[3][k]=file04_array[i][j]-file_mean[i][j];

                       k++;
                    }

       }


       // loops for the multiplication matrix (c_array) (4x4) of b_transpose and b. (4x400 x 400x4)

       double c_array[no_files][no_files];

        for(i=0;i<no_files;i++)
            {
                for(j=0;j<no_files;j++)
                    {
                        c_array[i][j]=0;
                            for(k=0;k<400;k++)
                                    {
                                        c_array[i][j]=c_array[i][j]+(b_transpose[i][k]*b[k][j]);
                                    }
                    }
            }


        Mat c;
        c = Mat(no_files, no_files, CV_64FC1, &c_array);

        // the following code finds the eigenvalues and eigenvectors and stores these value to eigenvalues and v respectively (4x4)
        Mat eigenvalues;
        Mat v;
        eigen(c,eigenvalues,v,-1,1);

        //finding transpose
        Mat v_trans;
        transpose(v,v_trans);

        //converting into 2d array
        double v_trans_array[no_files][no_files];
        for(i=0;i<no_files;i++)
            {
                for(j=0;j<no_files;j++)
                    {
                       v_trans_array[i][j]=v_trans.at<double>(i, j);
                    }
            }



        // loops for the multiplication of b (400x4) and v_trans_array(4x4) output: 400x4
        double u_array[no_files][no_files];

            for(i=0;i<i_horiz*i_vert;i++)
                {
                    for(j=0;j<no_files;j++)
                        {
                            u_array[i][j]=0;
                            for(k=0;k<no_files;k++)
                                    {
                                        u_array[i][j]=u_array[i][j]+(b[i][k]*v_trans_array[k][j]);
                                    }
                        }
                }

            // finding transpose (4x400)
            double u_trans[no_files][i_horiz*i_vert];

            for(i = 0; i < i_horiz*i_vert; ++i)
                    for(j = 0; j < no_files; ++j)
                        {
                            u_trans[j][i]=u_array[i][j];
                        }


           // loops for the multiplication of u_trans and b. (4x400 x 400x4) (4x4)

            double w_array[no_files][no_files];

            for(i=0;i<no_files;i++)
                    {
                        for(j=0;j<no_files;j++)
                            {
                                w_array[i][j]=0;
                                    for(k=0;k<i_horiz*i_vert;k++)
                                            {
                                                w_array[i][j]=w_array[i][j]+(u_trans[i][k]*b[k][j]);
                                            }
                            }
                    }

            //comparing input file
            string FilePath;
            cout << "Please enter the image file path to compare: ";
            cin >> FilePath;

            Mat readimage;
            readimage=imread(FilePath,IMREAD_GRAYSCALE);

            //storing the image vlaues in image array
                double readimage_array[i_horiz][i_vert];
                for(i=0;i<i_horiz;i++)
                    {
                        for(j=0;j<i_vert;j++)
                            {
                               readimage_array[i][j]=readimage.at<uchar>(i, j);

                            }
                    }

                // loops to find single column matrix l. (400x1)
                k=0;
                double l[i_horiz*i_vert][1];

                for(i=0;i<i_horiz;i++)
                        {
                            for(j=0;j<i_vert;j++)

                                {
                                   l[k][0]=readimage_array[i][j];
                                   k++;
                                }
                       }


                // loops for the multiplication of u_trans and l. (4x400 x 400x1) (4x1)

                double w1_array[no_files][1];

                    for(i=0;i<no_files;i++)
                                {
                                    for(j=0;j<1;j++)
                                        {
                                            w1_array[i][j]=0;
                                                for(k=0;k<i_horiz*i_vert;k++)
                                                        {
                                                            w1_array[i][j]=w1_array[i][j]+(u_trans[i][k]*l[k][j]);
                                                        }


                                        }
                                }


              //finding distance between w1 array elements and w array using vector formula
                        double distance[4];
                        printf("\n");

                        for(i=0;i<no_files;i++)
                        {
                            distance[i]=sqrt(pow(w1_array[0][0]-w_array[0][i],2)+pow(w1_array[1][0]-w_array[1][i],2)+pow(w1_array[2][0]-w_array[2][i],2)+pow(w1_array[3][0]-w_array[3][i],2));
                            printf("Distance between given image and database image file %d is: %7.0lf \n",i+1, distance[i]);

                        }

    return 0;
}

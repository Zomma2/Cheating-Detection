import sys 
sys.path.append('/mnt/acess/dfi/')
import dataframe_image as dfi
sys.path.append('/mnt/acess/')
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from datetime import datetime
from pytz import timezone
import numpy as np 
import cv2 
import webrtcvad
import collections
import contextlib
import sys
import wave
import numpy as np 
from pydub import AudioSegment
vad = webrtcvad.Vad(3)
import matplotlib.pyplot as plt
from pywaffle import Waffle



now = datetime.now(timezone('Egypt'))

class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        
def create_pose_df (arr1 , arr2) : 
  images_name = ['im '+str(i) for i in range(1,len(arr2)+1)]
  df = pd.DataFrame([arr1,arr2] , index = ['Horizontal' , 'Vertical'], columns = images_name).T
  return df 
  
def is_palgarsim_df (score):
  if score > 0 : 
    return 'Cheating' 
  else :
    return 'Non-Cheating'

def create_plots_headpose(df , path ) : 
  fig, axes = plt.subplots(nrows=4, ncols=2,figsize=(10,15))
  df['Horizontal'].plot(ax = axes[0,0],ylabel = 'Score' , title = 'Horizontal Pose Score line chart') ; 
  df['Horizontal'].plot(kind = 'bar' , ylabel = 'Score',ax = axes[0,1]
                        , title = 'Horizontal Pose Score bar chart') ; 

  df['Vertical'].plot(ax = axes[1,0],ylabel = 'Score', title = 'Vertical Pose Score bar chart') ; 
  df['Vertical'].plot(kind = 'bar' , ylabel = 'Score',ax = axes[1,1], title = 'Vertical Pose Score bar chart') ; 
 
  df['Vertical'].apply(is_palgarsim_df).value_counts().plot(kind = 'pie', autopct='%1.1f%%',ax = axes[2,0], ylabel='' , title = 'Vertical Pose Score pie chart') ; 
  df['Horizontal'].apply(is_palgarsim_df).value_counts().plot(kind = 'pie', autopct='%1.1f%%',ax = axes[2,1], ylabel='' ,title = 'Horizontal Pose Score pie chart') ; 


  df['Horizontal'].plot(ax = axes[3,0],ylabel = 'Score',title = 'Combined Pose Score line chart') ; 
  df['Vertical'].plot( ylabel = 'Score',ax = axes[3,0]) ; 
  sns.heatmap(df,cmap="GnBu",ax = axes[3,1]);
  axes[3,1].set_title('Combined Pose Score Heatmap chart') 


  plt.tight_layout()

  plt.savefig(path+'/head_pose.png')

def get_most_cheating_images(cheating_pose) : 
  sorted_mylist = sorted(((v, i) for i, v in enumerate(cheating_pose)), reverse=True)
  indices = [sorted_mylist[i][1] for i in range(4)]
  return indices
  
def images_to_report(image_list , indices) : 
  return [image_list[i] for i in indices ]


def create_images_subplot (images_to_report_list,path) : 
  size = len(images_to_report_list)

  global fig_img
  global axes_img
  fig_img, axes_img = plt.subplots(nrows=2, ncols=2,figsize=(10,15))
  i = 0
  j =0 
  for img in images_to_report_list : 
    if j > 1 : 
      i = i + 1 
      j = 0
    read_img = cv2.imread(img,cv2.IMREAD_COLOR)
    read_img=read_img[:,:,::-1]
    axes_img[i,j].imshow(read_img)
    j = j +1
    plt.tight_layout()
    plt.savefig(path+'/stud_photos.png',bbox_inches='tight')
    
def create_pdf_assets(student_image_list, H_pose_list , V_pose_list,path): 
  df = create_pose_df(H_pose_list , V_pose_list)
  create_plots_headpose(df,path)
  indices = get_most_cheating_images(H_pose_list)
  images_to_report_list = images_to_report(student_image_list , indices)
  create_images_subplot(images_to_report_list,path)



def df_to_image(df , path ) : 
  df_styled = df.style.background_gradient()
  dfi.export(df_styled, path+'/df_styled.png', table_conversion='matplotlib')
  return path+'/df_styled.png'


def export_to_pdf (df , student_name , stud_path , name = 'report.pdf') : 
  pdf = PDF('P', 'mm', 'A4')
  pdf.add_page()
  pdf.alias_nb_pages()
  pdf.image('/mnt/acess/Logo.jpg',w=200,h=200)
  pdf.set_font('Arial', '', 24)  
  pdf.ln(20)
  pdf.write(5, "Student {} Cheating Report".format(student_name))
  pdf.ln(10)
  pdf.set_font('Arial', '', 16)
  pdf.write(4, 'Reporting Time  : {}'.format(now))
  pdf.ln(5)
  pdf.add_page()
  pdf.set_font('Arial', '', 30)  
  pdf.write(10, "Report Overview" , 'C')
  pdf.ln(20)
  pdf.image(df_to_image(df,stud_path) ,w = 180 , h =100)
  pdf.add_page()
  pdf.write(10, "Considerable Images",'C')
  pdf.ln(15)
  pdf.image(stud_path+'/stud_photos.png',h=240,w=190)
  pdf.add_page()
  pdf.set_font('Arial', '', 24) 
  pdf.write(10, "Head Pose Estimator Graphical Overview",'C')
  pdf.ln(10)
  pdf.write(7, "(Only Considered on online Exams)",'C')
  pdf.ln(10)
  pdf.image(stud_path+'/head_pose.png',h=240,w=170)
  pdf.add_page()
  pdf.set_font('Arial', '', 24) 
  pdf.write(10, "Voice Activity Detection",'C')
  pdf.ln(10)
  pdf.image(stud_path+'/waffleVAD.png',w=190)
  pdf.output(stud_path + '/'+name, 'F')



def wafplot_VAD(nad_mi,path) : 
  data = {'Human Voice': round(nad_mi.sum()/nad_mi.shape[0],2), 'Non-Voice': round((nad_mi.shape[0]-nad_mi.sum())/nad_mi.shape[0],2) }
  fig = plt.figure(
      FigureClass=Waffle, 
      rows=10, 
      columns=40, 
      values=data, 
      title={'label': 'Voice Activity Percentage ', 'loc': 'left'},
      labels=["{0} ({1}%)".format(k, round(v*100)) for k, v in data.items()],
      legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0}
      ,figsize=(10,15)
  )
  print(path)
  plt.savefig(path+'waffleVAD.png',bbox_inches='tight')  
  
  
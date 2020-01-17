import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import random
#from libtiff import TIFF
from .data import readname

def str2point(point):
    rows = len(point)
    res = np.zeros((rows,2),dtype='int32')
    
    for idx in range(rows):
        res[idx,0], res[idx,1] = point[idx].split(',')
    return res
"""
返回所有表顶点每个点的x,y 坐标
"""

def isSorted(l,isAcs = True):#列表元素是否升序排序
    if isAcs:
        return all(l[i] <= l[i+1] for i in range(len(l)-1))
    #all函数返回true或Flase
    else:
        return all(l[i] >= l[i+1] for i in range(len(l)-1))
def isAlign(l): #所有元素差值不超过30
    return all(abs(l[i]-l[i+1])<=30 for i in range(len(l)-1))

def extractPoint(table):
    table_point = table.find('Coords').get('points').split()
    table_point = str2point(table_point)
    
    cell_point = []
    cell_loc = []
    for t in table.iter('cell'):
        tmp = t.find('Coords').get('points').split()
        cell_point.append(str2point(tmp))
        
        sr = int(t.get('start-row'))
        er = int(t.get('end-row'))
        sc = int(t.get('start-col'))
        ec = int(t.get('end-col'))
        cell_loc.append(np.array([sr,er,sc,ec]))
    
    return table_point, cell_point, cell_loc

def checkPoint(point, row, col):
    N = point.shape[0] #该cell边界坐标点的数目
    flag = True
    if  N == 2*(row+col):
        vline = point[0:row+1,:]
        #point存储的边的顺序是左边，下边，右边，上边，因此0-row是垂直线
        #因为边界具有一定的宽度，因此GT的边界数值会有几个像素的偏差
        flag1 = isSorted(vline[:,1]) and isAlign(vline[:,0])
        #两个条件都满足flag1为true。第一个是检查垂直边纵坐标递增，
        #因为边界像素偏差，第二个是检查横坐标差距不大
        hline = point[row:col+row+1,:]
        #此为下边，水平线
        flag2 = isSorted(hline[:,0]) and isAlign(hline[:,1])
        #检查水平边横坐标递增，纵坐标差距不大
        vline = point[col+row:col+2*row+1,:]
        flag3 = isSorted(vline[:,1], False) and isAlign(vline[:,0])
        hline = point[col+2*row:N,:]
        flag4 = isSorted(hline[:,0], False) and isAlign(hline[:,1])
        if not(flag1 and flag2 and flag3 and flag4):
            #四个标志位均成立时这些point组成的cell才算有效
            flag = False
    else:
        flag = False
    return flag



def preprocess(img_path, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rows,cols,ch = img.shape

    img_table = []
    fm_table = []
    
    for table in root.iter('table'):
        table_point, cell_point ,cell_loc = extractPoint(table)
        #cell_point所以cell的坐标点
        assert len(table_point)>=4, "table point less than 4"
        
        
        idx1 = np.argmin(np.array([x[1]+x[0] for x in table_point])) # top left
        idx2 = np.argmax(np.array([x[1]-x[0] for x in table_point])) # down left
        idx3 = np.argmax(np.array([x[1]+x[0] for x in table_point]))  # down right
        idx4 = np.argmin(np.array([x[1]-x[0] for x in table_point]))  # top right
        #根据四个点坐标找出表的上下左右点标注索引
        tmp = [table_point[idx1], table_point[idx2], table_point[idx3], table_point[idx4]]
        table_point = tmp
        
        table_row = table_point[1][1] - table_point[0][1]
        table_col = table_point[3][0] - table_point[0][0]

        #trans ormation
        pts1 = np.float32(table_point)
        pts2 = np.float32([[0, 0],[0, table_row],[table_col, table_row],[table_col, 0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        #仿射变换，将坐标点映射到以坐标原点为左下点的四边形的映射矩阵

        #label
        horizon_map = np.zeros((rows,cols) ,dtype="uint8")
        vertical_map = np.zeros((rows,cols) ,dtype="uint8")

        assert len(cell_loc)==len(cell_point)
            
        
        for idx in range(len(cell_point)):
            
            cell_row = cell_loc[idx][1] - cell_loc[idx][0] + 1 #cell width
            cell_col = cell_loc[idx][3] - cell_loc[idx][2] + 1 #cell length
            N = cell_point[idx].shape[0]
            #cell_point shape（坐标点数*2）
            if checkPoint(cell_point[idx], cell_row, cell_col):          
                assert N==2*(cell_row+cell_col), cell_col               
                # horizontal line feature map
                for i in range(cell_row, cell_row+cell_col):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)
                    # visuable horizontal line，最下边的水平线
                for i in range(1, cell_row):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][N-i-cell_col]), 2, 20)
                    # unvisuable horizontal line，除上和下边界之外的水平线
                for i in range(cell_col+2*cell_row, N-1):
                    cv2.line(horizon_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)
                    # visuable horizontal line 最顶端的水平线
                cv2.line(horizon_map, tuple(cell_point[idx][N-1]), tuple(cell_point[idx][0]), 1, 20)
                #最后一个点和第一个点连接起来

                # vertical line feature map
                for i in range(cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)
                    # visuable vertical line，最左端垂直线
                for i in range(cell_row+1, cell_col+cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][N-i+cell_row]), 2, 20)
                    # unvisuable vertical line 除左右边界之外的垂直线
                for i in range(cell_col+cell_row, cell_col+2*cell_row):
                    cv2.line(vertical_map, tuple(cell_point[idx][i]), tuple(cell_point[idx][i+1]), 1, 20)
                    # visuable vertical line 最右端边界线
            else:
                print("reverse")
                #对于不是正规四边形来说
#                 print("%d\t%d\t%d\t%d" %(idx,N,cell_row,cell_col))
                idx1 = np.argmax(np.array([x[1]-x[0] for x in cell_point[idx]]))  # down left
                idx2 = np.argmax(np.array([x[1]+x[0] for x in cell_point[idx]]))  # down right
                idx3 = np.argmin(np.array([x[1]-x[0] for x in cell_point[idx]]))  # top right
               # print(idx1, idx2, idx3)
#                 print(cell_point[idx])
#                 print("%d\t%d\t%d" %(idx1,idx2,idx3))
                #看cell是顺时针排列还是逆时针，顺时针需要反过来
                if idx1>idx2:
                    cell=[]
                    for i in cell_point[idx][1:]:
                        cell.append(list(i))
                    cell.append(list(cell_point[idx][0]))
                    cell.reverse()
                    idx1 = N-idx1-1
                    idx2 = N-idx2-1
                    idx3 = N-idx3-1
                   # print(idx1,idx2,idx3)

                else:
                    cell = cell_point[idx]

                # print(N)
                # print(cell)
                assert N==len(cell), "%d\t%d" %(N, len(cell))
                
                # horizontal line feature map
                for i in range(idx1, idx2):
                    cv2.line(horizon_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)
                    # visuable horizontal line，最下方水平线
                for i in range(idx3, N-1):
                    cv2.line(horizon_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)
                    # visuable horizontal line，上方水平线
                cv2.line(horizon_map, tuple(cell[N-1]), tuple(cell[0]), 1, 20)

                # vertical line feature map
                for i in range(idx1):
                    cv2.line(vertical_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable vertical line
                for i in range(idx2, idx3):
                    #print(i)
                    cv2.line(vertical_map, tuple(cell[i]), tuple(cell[i+1]), 1, 20)           # visuable vertical line

            
                flag1 = (idx1==cell_row)
                flag2 = ((idx2-idx1)==cell_col)
                flag3 = ((idx3-idx2)==cell_row)
                flag4 = ((N-idx3)==cell_col)
                
                 
                if flag1 and flag3:#若左右边界相等
                    for i in range(1, idx1):
                        cv2.line(horizon_map, tuple(cell[i]), tuple(cell[idx3-i]), 2, 20)  # unvisuable horizontal line
                else:#若左右边界长度不等
                    if flag1:
                        for i in range(1, idx1):
                            cv2.line(horizon_map, tuple(cell[i]), (cell[idx3][0],cell[i][1]), 2, 20)
                            # unvisuable horizontal line 以最左边行号为起点画水平线
                    elif flag3:
                        for i in range(idx2+1, idx3):
                            cv2.line(horizon_map, tuple(cell[i]), (cell[idx1][0],cell[i][1]), 2, 20)
                            # unvisuable horizontal line 以最右边行号为起点画水平线
                
                if flag2 and flag4:
                    for i in range(idx1+1, idx2):
                        cv2.line(vertical_map, tuple(cell[i]), tuple(cell[N-i+idx1]), 2, 20)  # unvisuable vertical line
                else:
                    if flag2:
                        for i in range(idx1+1, idx2):
                            cv2.line(vertical_map, tuple(cell[i]), (cell[i][0], cell[0][1]), 2, 20)  # unvisuable vertical line
                    elif flag4:
                        for i in range(idx3+1, N-1):
                            cv2.line(vertical_map, tuple(cell[i]), (cell[i][0], cell[idx1][1]), 2, 20)  # unvisuable vertical line
            
#                 cv2.line(vertical_map, tuple(cell_point[idx][0]), tuple(cell_point[idx][idx1]), 250, 20)
#                 cv2.line(horizon_map, tuple(cell_point[idx][idx1]), tuple(cell_point[idx][idx2]), 250, 20)
#                 cv2.line(vertical_map, tuple(cell_point[idx][idx2]), tuple(cell_point[idx][idx3]), 250, 20)
#                 cv2.line(horizon_map, tuple(cell_point[idx][idx3]), tuple(cell_point[idx][0]), 250, 20)

        img_tmp = cv2.warpPerspective(img,M,(table_col,table_row))
        img_table.append(img_tmp)
        #将图表做仿射变换，将图标区域平移到坐标轴
        
        horizon_map = cv2.warpPerspective(horizon_map,M,(table_col,table_row))
        vertical_map = cv2.warpPerspective(vertical_map,M,(table_col,table_row))
        fm_table.append((horizon_map,vertical_map))
        
    #函数返回图标图像和水平线图和垂直线图
    return img_table, fm_table

def write_filename(file_path, image_name):
    file = open(file_path,'a')
    for i in range(len(image_name)):
        s = str(image_name[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()


def split_train_val(dir_path, filename, val_percent=0.05):
    dataset = readname(os.path.join(dir_path,filename))
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    
    train_path = os.path.join(dir_path,'train.txt')
    test_path = os.path.join(dir_path, 'test.txt')
    
    write_filename(train_path, dataset[:-n])
    write_filename(test_path, dataset[-n:])
    
def read_train_val(dir_path):
    train_img = readname(os.path.join(dir_path,'train.txt'))
    test_img = readname(os.path.join(dir_path,'test.txt'))
    
    return {'train': train_img, 'val':test_img}

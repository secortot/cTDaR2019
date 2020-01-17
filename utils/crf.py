import numpy as np
import pydensecrf.densecrf as dcrf
#https://blog.csdn.net/weixin_33967071/article/details/94236827
#关于pydensecrf

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
   # U = U.transpose(2, 0, 1)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)
    #创建一元势

    d.addPairwiseGaussian(sxy=20, compat=3)
    #d.addPairwiseGaussian这个函数创建的是颜色无关特征，
    # 这里只有位置特征(只有参数实际相对距离sxy)，并添加到CRF中
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    #d.addPairwiseBilateral这个函数根据原始图像img
    # 创建颜色相关和位置相关特征并添加到CRF中，特征为(x,y,r,g,b)
    Q = d.inference(5)#5次迭代
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

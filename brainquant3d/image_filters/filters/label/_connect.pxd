from libc.string cimport memcpy

cdef extern from "opencv2/core.hpp":
  cdef int CV_8UC1
  cdef int CV_32S

cdef extern from "opencv2/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data

cdef extern from "opencv2/imgproc.hpp" namespace "cv":
  cdef int connectedComponents(Mat, Mat)

cdef inline void np2cvMat(unsigned char[:,::1] arr, Mat& out):
  cdef unsigned char* im_buff = &arr[0,0]
  cdef int r = arr.shape[0]
  cdef int c = arr.shape[1]
  out.create(r, c, CV_8UC1)
  memcpy(out.data, im_buff, r*c*1)

cdef inline void cvMat2np(Mat& arr, int[:,::1] out):
  cdef int* out_buff = &out[0,0]
  cdef int r = out.shape[0]
  cdef int c = out.shape[1]
  memcpy(out_buff, arr.data, r*c*4) # Rows x Columns x Size of variable type

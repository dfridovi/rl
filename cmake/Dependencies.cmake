include(ExternalProject)
set(rl_LIBRARIES "")

# Find Eigen.
find_package( Eigen3 REQUIRED )
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
list(APPEND rl_LIBRARIES ${EIGEN3_LIBRARIES})

# Find Google-gflags.
#include("cmake/External/gflags.cmake")
include("cmake/Modules/FindGflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND rl_LIBRARIES ${GFLAGS_LIBRARIES})

# Find Google-glog.
#include("cmake/External/glog.cmake")
include("cmake/Modules/FindGlog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND rl_LIBRARIES ${GLOG_LIBRARIES})

# Find OpenGL.
find_package( OpenGL REQUIRED )
include_directories(SYSTEM ${OPENGL_INCLUDE_DIRS})
list(APPEND rl_LIBRARIES ${OPENGL_LIBRARIES})

# Find GLUT.
find_package( GLUT REQUIRED )
include_directories(SYSTEM ${GLUT_INCLUDE_DIRS})
list(APPEND rl_LIBRARIES ${GLUT_LIBRARIES})

# Find Boost functional module.
find_package( Boost REQUIRED )
include_directories(SYSTEM ${BOOST_INCLUDE_DIRS})
list(APPEND rl_LIBRARIES ${BOOST_LIBRARIES})

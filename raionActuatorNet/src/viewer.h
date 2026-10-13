#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>
// Render-only native GLFW viewer. Mouse camera only; no forces or model edits.
struct Viewer {
  GLFWwindow* window=nullptr;
  mjvCamera camera; mjvOption option; mjvScene scene; mjrContext context;
  std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
  double t0=0,next=0,speed=1,lastx=0,lasty=0;
  bool pacing=false;
  Viewer(mjModel* m,bool enabled,bool realtime,double rate):speed(rate),pacing(enabled||realtime) {
    if(!enabled) return;
    if(!glfwInit()) throw std::runtime_error("GLFW initialization failed; use a graphical desktop");
    window=glfwCreateWindow(1200,800,"Raion ActuatorNet | native MuJoCo",nullptr,nullptr);
    if(!window) {glfwTerminate();throw std::runtime_error("cannot create viewer");}
    glfwMakeContextCurrent(window);glfwSwapInterval(0);
    mjv_defaultCamera(&camera);mjv_defaultOption(&option);mjv_defaultScene(&scene);mjr_defaultContext(&context);
    mjv_makeScene(m,&scene,10000);mjr_makeContext(m,&context,mjFONTSCALE_150);
    mju_copy3(camera.lookat,m->stat.center);camera.distance=std::max(1.,2.5*m->stat.extent);
    camera.azimuth=135;camera.elevation=-20;
    glfwSetWindowUserPointer(window,this);
    glfwSetScrollCallback(window,[](GLFWwindow* w,double,double y){
      auto v=static_cast<Viewer*>(glfwGetWindowUserPointer(w));v->camera.distance*=std::exp(-.1*y);
    });
  }
  void start(double time){t0=time;next=time;started=std::chrono::steady_clock::now();}
  void update(mjModel* m,mjData* d){
    if(window && glfwWindowShouldClose(window)) {close();pacing=false;}
    if(d->time<next) return;
    if(pacing) std::this_thread::sleep_until(started+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>((d->time-t0)/speed)));
    next=d->time+speed/60.;
    if(!window) return;
    double x,y;glfwGetCursorPos(window,&x,&y);
    if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS){camera.azimuth-=.3*(x-lastx);camera.elevation=std::clamp(camera.elevation-.3*(y-lasty),-89.,89.);}
    lastx=x;lasty=y;
    int width,height;glfwGetFramebufferSize(window,&width,&height);
    mjrRect viewport={0,0,width,height};
    mjv_updateScene(m,d,&option,nullptr,&camera,mjCAT_ALL,&scene);
    mjr_render(viewport,&scene,&context);
    glfwSwapBuffers(window);glfwPollEvents();
  }
  void close(){if(window){mjr_freeContext(&context);mjv_freeScene(&scene);glfwDestroyWindow(window);window=nullptr;glfwTerminate();}}
  ~Viewer(){close();}
};

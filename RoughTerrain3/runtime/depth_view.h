#ifndef ROUGHTERRAIN3_RUNTIME_DEPTH_VIEW_H_
#define ROUGHTERRAIN3_RUNTIME_DEPTH_VIEW_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

namespace roughterrain3 {

// Independent camera, scene, and GL context: main-view controls do not alter depth.
class DepthView {
 public:
  DepthView() {
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    mjv_defaultCamera(&camera_);
    mjv_defaultOption(&options_);
    options_.geomgroup[3] = 0;
    options_.sitegroup[0] = options_.sitegroup[1] = options_.sitegroup[2] = 0;
  }
  ~DepthView() { Close(); }
  DepthView(const DepthView&) = delete;
  DepthView& operator=(const DepthView&) = delete;

  void Close() {
    if (!window_) return;
    GLFWwindow* previous = glfwGetCurrentContext();
    glfwMakeContextCurrent(window_);
    mjr_freeContext(&context_);
    mjv_freeScene(&scene_);
    glfwDestroyWindow(window_);
    glfwMakeContextCurrent(previous == window_ ? nullptr : previous);
    window_ = nullptr;
  }

  void Update(const mjModel* model, mjData* data, bool& enabled) {
    if (window_ && glfwWindowShouldClose(window_)) {
      enabled = false;
      glfwSetWindowShouldClose(window_, GLFW_FALSE);
    }
    if (!enabled) {
      if (window_) glfwHideWindow(window_);
      return;
    }
    if (!model->ncam) {
      std::fprintf(stderr, "[GREENY] Depth view requires an XML camera.\n");
      enabled = false;
      return;
    }
    GLFWwindow* previous = glfwGetCurrentContext();
    if (!window_) {
      window_ = glfwCreateWindow(kWidth, kHeight, "GREENY Depth", nullptr, nullptr);
      if (!window_) throw std::runtime_error("failed to create depth window");
      glfwSetWindowUserPointer(window_, this);
      glfwSetKeyCallback(window_, Key);
      glfwMakeContextCurrent(window_);
      glfwSwapInterval(0);
      mjv_makeScene(model, &scene_, 2000);
      mjr_makeContext(model, &context_, mjFONTSCALE_100);
      mjr_resizeOffscreen(kWidth, kHeight, &context_);
      context_.readDepthMap = mjDEPTH_ZEROFAR;
      camera_.type = mjCAMERA_FIXED;
      camera_.fixedcamid = mj_name2id(model, mjOBJ_CAMERA, "cam_a");
      if (camera_.fixedcamid < 0) camera_.fixedcamid = 0;
      camera_count_ = model->ncam;
      depth_.resize(kWidth * kHeight);
      rgb_.resize(3 * kWidth * kHeight);
    }
    if (!glfwGetWindowAttrib(window_, GLFW_VISIBLE)) glfwShowWindow(window_);
    glfwMakeContextCurrent(window_);
    mjrRect viewport{0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
    if (viewport.width && viewport.height) {
      const mjrRect capture{0, 0, kWidth, kHeight};
      mjv_updateScene(model, data, &options_, nullptr, &camera_, mjCAT_ALL, &scene_);
      // Same depth-rendering flags used by MuJoCo's Python Renderer.
      scene_.flags[mjRND_SEGMENT] = 1;
      scene_.flags[mjRND_IDCOLOR] = 1;
      mjr_setBuffer(mjFB_OFFSCREEN, &context_);
      if (context_.currentBuffer != mjFB_OFFSCREEN)
        throw std::runtime_error("depth offscreen framebuffer unavailable");
      mjr_render(capture, &scene_, &context_);
      mjr_readPixels(nullptr, depth_.data(), capture, &context_);
      const float near = model->vis.map.znear * model->stat.extent;
      const float far = model->vis.map.zfar * model->stat.extent;
      // Match the float projection coefficients used by the renderer.
      const float c = -0.5f * (-(far + near) / (far - near)) - 0.5f;
      const float d = -0.5f * (-(2.f * far * near) / (far - near));
      for (int i = 0; i < kWidth * kHeight; ++i) {
        const double raw = depth_[i];
        const double metres = scene_.camera[0].orthographic ?
            far - raw * (far - near) : d / (raw + c);
        const bool valid = raw > 0 && std::isfinite(metres) && metres > 0;
        depth_[i] = valid ? metres : std::numeric_limits<float>::quiet_NaN();
        const double t = std::clamp(metres / range_, 0.0, 1.0);
        // Near red -> mid green -> far blue; no surface is black.
        rgb_[3*i] = valid ? 255 * std::max(0.0, 1.0 - 2*t) : 0;
        rgb_[3*i+1] = valid ? 255 * (1.0 - std::abs(2*t - 1.0)) : 0;
        rgb_[3*i+2] = valid ? 255 * std::max(0.0, 2*t - 1.0) : 0;
      }
      // mjr_drawPixels preserves GL state. Reset to 2D first, otherwise the
      // reversed-Z depth test left by mjr_render rejects the color image.
      mjr_rectangle(capture, 0, 0, 0, 1);
      mjr_drawPixels(rgb_.data(), nullptr, capture, &context_);
      // Letterbox when resized; buffers stay in OpenGL bottom-up order throughout.
      mjr_setBuffer(mjFB_WINDOW, &context_);
      mjr_rectangle(viewport, 0, 0, 0, 1);
      const double scale = std::min(viewport.width / double(kWidth),
                                    viewport.height / double(kHeight));
      mjrRect image{0, 0, int(kWidth * scale), int(kHeight * scale)};
      image.left = (viewport.width - image.width) / 2;
      image.bottom = (viewport.height - image.height) / 2;
      mjr_setBuffer(mjFB_OFFSCREEN, &context_);
      mjr_blitBuffer(capture, image, 1, 0, &context_);
      mjr_setBuffer(mjFB_WINDOW, &context_);
      const char* name = mj_id2name(model, mjOBJ_CAMERA, camera_.fixedcamid);
      const float center = depth_[(kHeight/2)*kWidth + kWidth/2];
      char info[384];
      std::snprintf(info, sizeof(info),
          "%s | 640x480 | center Z: %.3f m\n"
          "Red: near  Green: %.1f m  Blue: >= %.1f m  Black: no surface\n"
          "Tab: camera  -/+: range  F12/Esc: close",
          name ? name : "camera", center, range_/2, range_);
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "DEPTH", info, &context_);
      glfwSwapBuffers(window_);
    }
    glfwMakeContextCurrent(previous);
  }

 private:
  static void Key(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* self = static_cast<DepthView*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_F12)
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    else if (key == GLFW_KEY_TAB)
      self->camera_.fixedcamid = (self->camera_.fixedcamid + 1) % self->camera_count_;
    else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
      self->range_ = std::min(1000.0, self->range_ * 2);
    else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
      self->range_ = std::max(0.125, self->range_ / 2);
  }
  static constexpr int kWidth = 640, kHeight = 480;
  GLFWwindow* window_ = nullptr;
  mjvScene scene_{};
  mjrContext context_{};
  mjvCamera camera_{};
  mjvOption options_{};
  int camera_count_ = 0;
  double range_ = 5.0;
  std::vector<float> depth_;
  std::vector<unsigned char> rgb_;
};
}  // namespace roughterrain3
#endif  // ROUGHTERRAIN3_RUNTIME_DEPTH_VIEW_H_

"""实时相机视野窗口（MuJoCo 离屏渲染 + OpenCV 显示）。

用途：主窗口看 3D 场景的同时，另一个窗口实时显示"相机拍到什么"——
这是后续做视觉策略（模仿学习/强化学习）时的观测来源。

用法：
    cam = CameraView(model, data, cam_name="cam_top", width=800, height=600)
    cam.update()          # 渲染一帧并刷新窗口（内部 cv2.waitKey(1) 泵事件）
    cam.close()
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class CameraView:
    """单相机实时视野窗口。"""
    def __init__(self, model, data, cam_name: str = "cam_top", width: int = 800,
                 height: int = 600, window_title: Optional[str] = None,
                 overlay: bool = True, show_tcp: bool = True, every: int = 4):
        import cv2
        import mujoco
        self.cv2, self.mujoco = cv2, mujoco
        self.model, self.data = model, data
        self.width, self.height = width, height
        self.cam_name = cam_name
        # 注意：OpenCV 的窗口标题在 Windows 上不吃中文（会变乱码），因此用英文标题
        self.title = window_title or f"Camera View: {cam_name}"
        self.overlay = overlay
        self.show_tcp = show_tcp
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self._closed = False
        self.every = max(int(every), 1)     # 每隔多少次 sync 渲染一帧（控制开销）
        self._n = 0
        self._tcp_fun = None
        try:      # 让 OpenCV 窗口可以自由缩放
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, width, height)
        except Exception:
            pass

    def set_tcp_fn(self, fn) -> None:
        """可选：传入 `fn() -> (x, y, z)`，用于在画面上标出末端位置。"""
        self._tcp_fun = fn

    def _project(self, xyz: Sequence[float]):
        """把世界坐标投影到该相机的像素坐标（用 MuJoCo 的相机内参）。"""
        mujoco = self.mujoco
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if cid < 0:
            return None
        # 世界 → 相机坐标系 → 像素（相机朝自身 -z 看，fovy 给垂直视场）
        fovy = np.deg2rad(float(self.model.cam_fovy[cid]))
        pos = self.data.cam_xpos[cid]
        mat = self.data.cam_xmat[cid].reshape(3, 3)
        d = mat.T @ (np.asarray(xyz, dtype=float) - pos)      # 相机坐标系（-z 朝前）
        if d[2] >= -1e-6:
            return None
        f = 0.5 * self.height / np.tan(fovy / 2.0)
        u = self.width / 2.0 + f * d[0] / (-d[2])
        v = self.height / 2.0 - f * d[1] / (-d[2])
        return int(u), int(v)

    def update(self) -> None:
        if self._closed:
            return
        self._n += 1
        if self._n % self.every:
            return
        try:
            self.renderer.update_scene(self.data, camera=self.cam_name)
            frame = self.renderer.render()[:, :, ::-1].copy()      # RGB → BGR
        except Exception:
            return
        if self.overlay:
            cv2 = self.cv2
            cv2.putText(frame, f"{self.cam_name}  {self.width}x{self.height}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (0, 0), (self.width - 1, self.height - 1),
                          (200, 200, 200), 1)
            if self.show_tcp and self._tcp_fun is not None:
                try:
                    uv = self._project(self._tcp_fun())
                    if uv is not None:
                        cv2.drawMarker(frame, uv, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                except Exception:
                    pass
        try:
            self.cv2.imshow(self.title, frame)
            self.cv2.waitKey(1)
        except Exception:
            self._closed = True

    def close(self) -> None:
        if not self._closed:
            try:
                self.cv2.destroyWindow(self.title)
            except Exception:
                pass
            self._closed = True


OBJECT_PROBE = [(0.0, 0.0, 0.0), (0.012, 0.012, 0.012), (-0.012, 0.012, 0.012),
                (0.012, -0.012, 0.012), (-0.012, -0.012, 0.012),
                (0.0, 0.0, 0.015)]


def visible_from(model, data, cam_name: str, target: Sequence[float],
                 tol: float = 0.03, probes: Sequence[Sequence[float]] | None = None,
                 fov_size: tuple[int, int] | None = (640, 480)) -> bool:
    """从指定相机看目标是否**未被完全遮挡**（MuJoCo 多点射线检测）。

    用途：判断"物体是否在俯视相机里被机械臂挡住"——这正是双相机协同的触发条件。
    - 默认用 6 个探针点（中心 + 四个上角 + 顶点）：**任一条射线通畅即算可见**，
      这样"手指从两侧夹住、但顶部露出来"的情况不会被误判为遮挡；
    - 若给出 fov_size=(w,h)，还会检查目标是否落在画面内（线通不等于在视野里）；
    - tol 取"目标自身尺寸 + 余量"：射线会先打到目标表面而不是中心；
    - 相机的安装 body 会被排除（否则射线会撞到相机自己的支架/外壳）。
    """
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        return False
    body_exclude = int(model.cam_bodyid[cid])
    origin = data.cam_xpos[cid].copy()
    R = data.cam_xmat[cid].reshape(3, 3)
    fovy = np.deg2rad(float(model.cam_fovy[cid]))
    base = np.asarray(target, dtype=float)
    geomid = np.zeros(1, dtype=np.int32)
    for off in (probes if probes is not None else OBJECT_PROBE):
        p = base + np.asarray(off, dtype=float)
        if fov_size is not None:                     # 视场角检查
            w, h = fov_size
            d_cam = R.T @ (p - origin)
            if d_cam[2] >= -1e-6:
                continue
            f = 0.5 * h / np.tan(fovy / 2.0)
            u = w / 2.0 + f * d_cam[0] / (-d_cam[2])
            v = h / 2.0 - f * d_cam[1] / (-d_cam[2])
            if not (0 <= u < w and 0 <= v < h):
                continue
        vec = p - origin
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return True
        hit = mujoco.mj_ray(model, data, origin, vec / dist, None, 1,
                            body_exclude, geomid)
        if hit < 0 or hit > dist - tol:
            return True
    return False


class MultiCameraView:
    """多相机实时视野窗口（并排显示，用于"双相机协同"观察）。

    典型用法：`cam_top`（全局俯视）+ `cam_eye`（J4 支架上的第一人称），
    每帧并排渲染并标注每台相机对目标点的可见性（OK / BLOCKED）。
    """

    def __init__(self, model, data, cams: Sequence[str] = ("cam_top", "cam_eye"),
                 width: int = 640, height: int = 480,
                 window_title: str = "Camera Views (top | eye-in-hand)",
                 every: int = 4):
        import cv2
        import mujoco
        self.cv2, self.mujoco = cv2, mujoco
        self.model, self.data = model, data
        self.cams = list(cams)
        self.width, self.height = width, height
        self.title = window_title
        self.every = max(int(every), 1)
        self._n = 0
        self._closed = False
        self.renderers = [mujoco.Renderer(model, height=height, width=width)
                          for _ in self.cams]
        self._target_fn = None
        try:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, width * len(self.cams), height)
        except Exception:
            pass

    def set_target_fn(self, fn) -> None:
        """传入 `fn() -> (x,y,z)`：要在画面上标注可见性的目标点（比如物体中心）。"""
        self._target_fn = fn

    def update(self) -> None:
        if self._closed:
            return
        self._n += 1
        if self._n % self.every:
            return
        cv2 = self.cv2
        target = None
        if self._target_fn is not None:
            try:
                target = self._target_fn()
            except Exception:
                target = None
        tiles = []
        for cam, renderer in zip(self.cams, self.renderers):
            try:
                renderer.update_scene(self.data, camera=cam)
                img = renderer.render()[:, :, ::-1].copy()
            except Exception:
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            txt = cam
            color = (255, 255, 255)
            if target is not None:
                ok = visible_from(self.model, self.data, cam, target)
                txt = f"{cam}  {'VISIBLE' if ok else 'BLOCKED'}"
                color = (120, 255, 120) if ok else (80, 80, 255)
            cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (0, 0), (self.width - 1, self.height - 1), (200, 200, 200), 1)
            tiles.append(img)
        frame = np.hstack(tiles)
        try:
            cv2.imshow(self.title, frame)
            cv2.waitKey(1)
        except Exception:
            self._closed = True

    def close(self) -> None:
        if not self._closed:
            try:
                self.cv2.destroyWindow(self.title)
            except Exception:
                pass
            self._closed = True

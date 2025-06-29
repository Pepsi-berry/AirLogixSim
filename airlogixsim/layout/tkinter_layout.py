import tkinter as tk
import time
import math
import numpy as np
from .base_layout import BaseLayout
from ..utils.tk_utils import get_map_data_from_osm, parse_location_info, create_rotated_images, make_image_transparent, \
    clear_screen
from ..utils.my_generator import random_colors_generator
from PIL import Image, ImageDraw, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TkinterLayout(tk.Tk, BaseLayout):
    def __init__(self, config, env):
        tk.Tk.__init__(self)
        self._env = env
        # -------------load files and images----------------
        self.img_path = config['visualization']['icon_path']
        self.net_file_path = config['sumo']['sumo_net']
        self.osm_file_path = config['sumo']['sumo_osm']
        self.car_images = create_rotated_images(f'{self.img_path}/car.png', size=(20, 30))
        self.uav_image = make_image_transparent(f'{self.img_path}/uav.png', size=(30, 30))
        self.color_generator = random_colors_generator(20)
        self.simulation_delay = 0
        self.last_updated_time = time.time()

        # -------------read net.xml file to adjust the boundary of the map----------------
        self.canvas_width = 800
        self.canvas_height = 800
        self.scale_factor = 1
        self.calculate_bbox()
        _width = self.location_bound[2] - self.location_bound[0]
        _height = self.location_bound[3] - self.location_bound[1]
        self.uav_positions = []
        aspect_ratio = self.canvas_width / self.canvas_height
        if _width / _height > aspect_ratio:
            self.canvas_height = int(self.canvas_width * _height / _width)
        else:
            self.canvas_width = int(self.canvas_height * _width / _height)

        # -------------initialize the window event----------------
        self.title(f"AirLogixSim Platform - NaMI Lab, Tongji University: {env.airfogsim_label}")
        self.geometry(f"{self.canvas_width}x{self.canvas_height}")
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind('<ButtonPress-1>', self.drag_start)
        self.canvas.bind('<B1-Motion>', self.drag_move)
        self.buttons_frame = tk.Frame(self)
        self.map_grid_activate_button = tk.Button(self.buttons_frame, text="Activate Map Grid",
                                                  command=self.create_map_grid_window)
        self.map_grid_activate_button.pack(side=tk.LEFT)
        self.buttons_frame.pack(side=tk.BOTTOM)
        self.isShowMapGrid = False
        self.map_grid_window = None
        self.map_grid_figure = None
        self.canvas.pack()

        # -------------draw the map----------------
        self.map_data = get_map_data_from_osm(self.osm_file_path, self.bbox, self.proj_params, self.netOffset)
        self.map_image_data = self.draw_map(self.map_data)
        self.canvas_map_image = self.canvas.create_image(0, 0, image=self.map_image_data, anchor=tk.NW)

        self.canvas_vehicle_images = {}  # env.vehicleId -> canvas image
        self.canvas_uav_images = {}  # env.uavId -> canvas image
        self.canvas_dots = {}  # env.missionId -> canvas oval
        self.canvas_time_image = None

    def create_map_grid_window(self):
        self.isShowMapGrid = not self.isShowMapGrid
        if self.isShowMapGrid:
            self.map_grid_activate_button.config(text="Deactivate Map Grid")
            if self.map_grid_window is None or not self.map_grid_window.winfo_exists():
                self.map_grid_window = tk.Toplevel(self)
                self.map_grid_window.title("Map Grid")
                self.map_grid_window.geometry(f"400x400+{self.canvas_width}+0")
                # before map_grid window distroyed, the map_grid_activate_button should be set to "Deactivate Map Grid"
                self.map_grid_window.protocol("WM_DELETE_WINDOW", self.create_map_grid_window)
        else:
            self.map_grid_activate_button.config(text="Activate Map Grid")
            # remove ax attribute to reinitialize the figure
            del self.ax
            if self.map_grid_window is not None and self.map_grid_window.winfo_exists():
                self.map_grid_window.destroy()

    def render(self):
        self.update_uav_position()
        self.update_vehicle_position_and_direction()
        self.update_visualization()
        self.update()
        if self._env.simulation_time % 10 == 0:
            self.canvas_vehicle_images = {}
            self.canvas.delete("vehicle")

    def update_visualization(self):
        self.draw_canvas_images(self.canvas_vehicle_images, self.vehicle_positions_in_pixel, self.car_images, "vehicle",
                                directions=self.vehicle_directions)
        self.draw_canvas_images(self.canvas_uav_images, self.uav_positions_in_pixel, [self.uav_image], "uav")
        self.draw_canvas_dots(self.canvas_dots, self.mission_positions_in_pixel, 'mission')
        self.draw_time()
        if self.isShowMapGrid and self.map_grid_window is not None and self.map_grid_window.winfo_exists():
            self.draw_map_grid()

    def draw_map_grid(self):
        veh_id_grid_map = self._env.traffic_manager.map_by_grid  # np.zeros((row_num, col_num), dtype=np.object)
        n_row = veh_id_grid_map.shape[0]
        n_col = veh_id_grid_map.shape[1]
        veh_number_grid_map = np.vectorize(len)(veh_id_grid_map)
        # heatmap
        if not hasattr(self, 'ax'):
            self.fig = Figure(figsize=(5, 5), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.map_grid_figure = FigureCanvasTkAgg(self.fig, master=self.map_grid_window)
            self.map_grid_figure.get_tk_widget().pack()
        else:
            self.colorbar.remove()
            self.ax.cla()
        cax = self.ax.matshow(veh_number_grid_map, cmap='hot')
        self.colorbar = self.fig.colorbar(cax)
        x_range = np.arange(n_col, step=5)
        y_range = np.arange(n_row, step=5)
        x_label = np.arange(n_col * self._env.traffic_manager.grid_width, step=5 * self._env.traffic_manager.grid_width)
        y_label = np.arange(n_row * self._env.traffic_manager.grid_width, step=5 * self._env.traffic_manager.grid_width)
        self.ax.set_xticks(x_range)
        self.ax.set_yticks(y_range)
        self.ax.set_xticklabels(x_label)
        self.ax.set_yticklabels(y_label)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.map_grid_figure.draw()

    def _get_position_by_id(self, id):
        if id in self.uav_positions_in_pixel:
            return self.uav_positions_in_pixel[id]
        if id in self.vehicle_positions_in_pixel:
            return self.vehicle_positions_in_pixel[id]
        return None

    def draw_time(self):
        if self.canvas_time_image is not None:
            self.canvas.delete(self.canvas_time_image)
        posx = 0.65 * self.canvas_width
        posy = 0.9 * self.canvas_height
        # simulation delay 只显示小数点后两位
        self.simulation_delay = time.time() - self.last_updated_time
        self.canvas_time_image = self.canvas.create_text(posx, posy,
                                                         text="Time: {} Vehicle Num: {} UAV Num: {} \n Step-wise Simulation Delay: {} ms".format(
                                                             self._env.simulation_time,
                                                             len(self._env.vehicle_ids_as_index),
                                                             len(self._env.uav_ids_as_index),
                                                             round(self.simulation_delay * 1000, 3)), anchor=tk.NW)
        self.last_updated_time = time.time()

    def draw_canvas_images(self, canvas_images: dict, positions: dict, image_set, tag, directions=None):
        to_delete_keys = canvas_images.keys() - positions.keys()
        for key in to_delete_keys:
            self.canvas.delete(canvas_images[key])
            del canvas_images[key]
        for key, position in positions.items():
            direction = directions[key] if directions is not None else 0
            rotated_image = image_set[(math.floor(direction / 10) - 9) % 36] if directions is not None else image_set[0]
            if key not in canvas_images:
                canvas_image = self.canvas.create_image(position[0], position[1], image=rotated_image, anchor=tk.CENTER,
                                                        tag=tag)
                canvas_images[key] = canvas_image
            else:
                self.canvas.move(canvas_images[key], position[0] - self.canvas.coords(canvas_images[key])[0],
                                 position[1] - self.canvas.coords(canvas_images[key])[1])

    def draw_canvas_dots(self, canvas_dots: dict, positions: dict, tag):
        to_delete_keys = canvas_dots.keys() - positions.keys()
        for key in to_delete_keys:
            self.canvas.delete(canvas_dots[key])
            del canvas_dots[key]
        for key, dot_positions in positions.items():
            for dot_position in dot_positions:
                if key not in canvas_dots:
                    canvas_dot = self.canvas.create_oval(dot_position[0], dot_position[1], dot_position[0],
                                                         dot_position[1], width=3, fill="red",outline='red')
                    canvas_dots[key] = []
                    canvas_dots[key].append(canvas_dot)

    def update_uav_position(self):
        uav_infos = EntityScheduler.getAllNodeInfos(self._env, ['uav'])
        self.uav_positions_in_pixel = {}
        for info_dict in uav_infos:
            x, y = info_dict['position_x'], info_dict['position_y']
            x, y = self.position_to_pixel(x, y)
            self.uav_positions_in_pixel[info_dict['id']] = [x, y]

    def update_vehicle_position_and_direction(self):
        vehicle_infos = EntityScheduler.getAllNodeInfos(self._env, ['vehicle'])
        self.vehicle_positions_in_pixel = {}
        self.vehicle_directions = {}
        for info_dict in vehicle_infos:
            x, y = info_dict['position_x'], info_dict['position_y']
            x, y = self.position_to_pixel(x, y)
            self.vehicle_positions_in_pixel[info_dict['id']] = [x, y]
            self.vehicle_directions[info_dict['id']] = info_dict['angle']

    def close(self):
        clear_screen()
        return super().close()

    # 拖拽开始函数
    def drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def zoom(self, event):
        # 滚轮事件
        # 判断滚轮向上还是向下
        if event.delta > 0:
            self.scale_factor *= 1.1
        elif event.delta < 0:
            self.scale_factor /= 1.1
        # check scale factor在 1, 3之间
        self.scale_factor = min(max(1, self.scale_factor), 3)
        # 根据 scale_factor 调整 创建图片的大小
        self.map_image_data = self.draw_map(self.map_data)
        self.canvas_map_image = self.canvas.create_image(0, 0, image=self.map_image_data, anchor=tk.NW)
        self.canvas_map_image = None
        self.canvas_uav_images = {}
        self.canvas.delete("uav")
        self.canvas_rsu_images = {}
        self.canvas.delete("rsu")
        self.canvas_vehicle_images = {}
        self.canvas.delete("vehicle")

    def drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def calculate_bbox(self):
        # 从net.xml文件中读取地图的bbox,通过parse_location_info函数
        conv_boundary, orig_boundary, proj_params, netOffset = parse_location_info(self.net_file_path)
        orig_boundary = tuple(map(float, orig_boundary.split(',')))
        conv_boundary = tuple(map(float, conv_boundary.split(',')))
        min_x = orig_boundary[0]
        min_y = orig_boundary[1]
        max_x = orig_boundary[2]
        max_y = orig_boundary[3]
        self.proj_params = proj_params
        self.netOffset = netOffset
        self.bbox = min_x, min_y, max_x, max_y
        self.location_bound = conv_boundary

    def draw_map(self, map_data):
        map_image = Image.new("RGBA",
                              (int(self.canvas_width * self.scale_factor), int(self.canvas_height * self.scale_factor)),
                              (255, 255, 255, 255))
        draw = ImageDraw.Draw(map_image)
        for road_segment in map_data:
            (x1, y1), (x2, y2) = road_segment
            # 判断x1, y1, x2, y2是否在 location_bound 中
            if x1 < self.location_bound[0] or x1 > self.location_bound[2] or y1 < self.location_bound[1] or y1 > \
                    self.location_bound[3] or x2 < self.location_bound[0] or x2 > self.location_bound[2] or y2 < \
                    self.location_bound[1] or y2 > self.location_bound[3]:
                continue

            # 先在 location_bound 进行归一化, 然后乘以 canvas_width, canvas_height，最后乘以 scale_factor
            x1, y1 = self.position_to_pixel(x1, y1, inverse=True)
            x2, y2 = self.position_to_pixel(x2, y2, inverse=True)
            draw.line([(x1, y1), (x2, y2)], fill="gray", width=1)
        return ImageTk.PhotoImage(map_image)

    def position_to_pixel(self, x, y, inverse=False):
        x = (x - self.location_bound[0]) / (self.location_bound[2] - self.location_bound[0])
        if inverse:
            y = (y - self.location_bound[1]) / (self.location_bound[3] - self.location_bound[1])
        else:
            y = (y - self.location_bound[1]) / (self.location_bound[3] - self.location_bound[1])
        x *= self.scale_factor * self.canvas_width
        y *= self.scale_factor * self.canvas_height
        return x, y

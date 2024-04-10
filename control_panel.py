import sys
import os
import time
import gc
import socket
from PyQt5 import QtGui
from pympler import asizeof
import pythoncom
import pyvisa
import pulse_odmr_ui
import pandas as pd
import numpy as np
import pyqtgraph as pg
import nidaqmx
from nidaqmx.constants import *
from nidaqmx.stream_readers import CounterReader
from threading import Thread, active_count
from ctypes import *
#import JSON-RPC Pulse Streamer wrapper class, to use Google-RPC import from pulsestreamer.grpc
from pulsestreamer import PulseStreamer, Sequence, OutputState, findPulseStreamers


from PyQt5.QtGui import QIcon, QPixmap, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsDropShadowEffect, QFileDialog, QDesktopWidget, QVBoxLayout

class ConfigureChannels():
    def __init__(self):
        super().__init__()
        self._pulser_channels = {
            'ch_aom': 0, # output channel 0: AOM control
            'ch_switch': 1, # output channel 1: MW switch control
            'ch_tagger': 2, # output channel 2 
            'ch_sync': 3, # output channel 3
            'ch_daq': 4, # NI gate channel
            'ch_mw_source': 5 # N5181A frequency change channel
        }
        self._timetagger_channels = {
            'click_channel': 1,
            'start_channel':2,
            'next_channel':-2,
            # 'sync_channel':tt.CHANNEL_UNUSED,
        }   
        self._ni_6363_channels = {
            'apd_channel':'/Dev2/PFI0',
            'clock_channel':'/Dev2/PFI1',
            'odmr_ctr_channel':'/Dev2/ctr0'
        } 
    @property
    def pulser_channels(self):
        return self._pulser_channels
    @property
    def timetagger_channels(self):
        return self._timetagger_channels
    @property
    def ni_6363_channels(self):
        return self._ni_6363_channels
    
class Hardware():
    def __init__(self):
        super().__init__()

    def pulser_generate(self):
        devices = findPulseStreamers()
        # print(devices)
        # DHCP is activated in factory settings
        if devices !=[]:
            ip = devices[0][0]
        else:
            # if discovery failed try to connect by the default hostname
            # IP address of the pulse streamer (default hostname is 'pulsestreamer')
            print("No Pulse Streamer found")

        #connect to the pulse streamer
        pulser = PulseStreamer(ip)

        # Print serial number and FPGA-ID

        return pulser
    def daq_task_generate(self, apd_channel, odmr_ctr_channel, **kwargs):
        task = nidaqmx.Task()
        channel = task.ci_channels.add_ci_count_edges_chan(
            counter=odmr_ctr_channel,
            edge=Edge.RISING,
            count_direction=CountDirection.COUNT_UP
        )
        channel.ci_count_edges_term = apd_channel
        channel.ci_count_edges_active_edge = Edge.RISING
        
        return task, channel
    
class MyWindow(pulse_odmr_ui.Ui_Form, QWidget):

    rf_info_msg = pyqtSignal(str)
    pulse_streamer_info_msg = pyqtSignal(str)
    data_processing_info_msg = pyqtSignal(str)
    odmr_data_info_msg = pyqtSignal(list)


    def __init__(self):

        super().__init__()

        # init UI
        self.setupUi(self)
        self.ui_width = int(QDesktopWidget().availableGeometry().size().width()*0.75)
        self.ui_height = int(QDesktopWidget().availableGeometry().size().height()*0.8)
        self.resize(self.ui_width, self.ui_height)
        center_pointer = QDesktopWidget().availableGeometry().center()
        x = center_pointer.x()
        y = center_pointer.y()
        old_x, old_y, width, height = self.frameGeometry().getRect()
        self.move(int(x - width / 2), int(y - height / 2))

        # set flag off and widget translucent
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # set window blur
        self.render_shadow()
        
        # init window button signal
        self.window_btn_signal()

        '''
        RF init
        '''
        # Init RF combobox ui
        self.rf_cbx_test()
        
        # Init RF setup info ui
        self.rf_info_ui()

        # Init RF signal
        self.my_rf_signal()
        '''
        Configure channels
        '''
        channel_config = ConfigureChannels()
        pulser_channels = channel_config.pulser_channels
        daq_channels = channel_config.ni_6363_channels
        self._channels = {**pulser_channels, **daq_channels}

        # print(self._channels)
        '''
        PULSER init
        '''
        self.hardware = Hardware()
        self.pulse_streamer_singal_init()
        self.pulse_streamer_info_ui()
        self.pulse_streamer_info_msg.connect(self.pulse_streamer_slot)
        # self.pulser_daq_on_activate()
        
        '''
        Data processing init
        '''
        self._data_container = []
        self.plot_ui_init()
        self.data_processing_signal()
        self.data_processing_info_ui()
        
    def data_processing_signal(self):

        # Message signal
        self.data_processing_info_msg.connect(self.data_processing_slot)
        self.odmr_data_info_msg.connect(self.plot_result)
        # Scroll area updating signal
        self.data_processing_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.data_processing_scroll.verticalScrollBar().setValue(
                self.data_processing_scroll.verticalScrollBar().maximum()
            )
        )
        # plot signal
        self.save_plot_data_btn.clicked.connect(self.save_plot_data)

        # infinite line signal
        # self.data_infinite_line.sigPositionChangeFinished.connect(self.reset_infinite_line_spbx_value)

    def save_plot_data(self):
        
        pass
        
    def plot_result(self, data):

        n_sample = self._pulse_configuration['n_sample']

        data_array = np.array(data)
        contrast_data = np.sum(data_array, axis=0,dtype=np.uint32)
        # print(len(time_data))
        # print(len(contrast_data))
        self.rabi_curve.setData(n_sample, contrast_data)   
                     
    def data_processing_info_ui(self):

        self.data_processing_msg.setWordWrap(True)  # 自动换行
        self.data_processing_msg.setAlignment(Qt.AlignTop)  # 靠上

        # # 用于存放消息
        self.data_processing_msg_history = []

    def data_processing_slot(self, msg):

        # print(msg)
        self.data_processing_msg_history.append(msg)
        self.data_processing_msg.setText("<br>".join(self.data_processing_msg_history))
        self.data_processing_msg.resize(700, self.data_processing_msg.frameSize().height() + 20)
        self.data_processing_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容
    
    def generate_infinite_line(self, pos=0, pen=None, label=None):

        line = pg.InfiniteLine(
                pos=pos, 
                angle=90, 
                pen=pen, 
                movable=True, 
                bounds=(0,None), 
                hoverPen=None, 
                label=label, 
                labelOpts={'position': 0.99}, 
                span=(0, 1), 
                markers=None, 
                name=None
            )
        return line
    def create_plot_widget(self, xlabel, ylabel, title, frame, infiniteLine=False):
        plot = pg.PlotWidget(enableAutoRange=True, useOpenGL=True)
        graph_widget_layout = QVBoxLayout()
        graph_widget_layout.addWidget(plot)
        frame.setLayout(graph_widget_layout)
        plot.setLabel("left", ylabel)
        plot.setLabel("bottom", xlabel)
        plot.setTitle(title, color='k')
        plot.setBackground(background=None)
        plot.getAxis('left').setPen('k')
        plot.getAxis('left').setTextPen('k')
        plot.getAxis('bottom').setPen('k')
        plot.getAxis('bottom').setTextPen('k')
        plot.getAxis('top').setPen('k')
        plot.getAxis('right').setPen('k')
        plot.showAxes(True)
        plot.showGrid(x=False, y=True)
        curve = plot.plot(pen=pg.mkPen(color=(255,85,48), width=2))        
        if infiniteLine == True:
            mw_start = int(self.mw_start_spbx.value())
            data_pen = pg.mkPen(color='b', width=1)
            self.data_infinite_line = self.generate_infinite_line(pen=data_pen,pos=mw_start,label='pos')
            plot.addItem(self.data_infinite_line)
        return curve
    def reset_infinite_line_spbx_value(self):

        data_pos = round(self.data_infinite_line.value())
        self.data_position_spbx.setValue(data_pos)

    def plot_ui_init(self):
        current_tab_index = self.pulse_tab.currentIndex()
        current_tab_name = self.pulse_tab.tabText(current_tab_index)
        self.rabi_curve = self.create_plot_widget(
            xlabel='Time (ns)',
            ylabel='Counts(a.u.)',
            title=current_tab_name,
            frame=self.rabi_graph_frame,
            infiniteLine=False
        )

    def pulse_streamer_singal_init(self):

        # ASG scroll area scrollbar signal
        self.pulse_streamer_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.pulse_streamer_scroll.verticalScrollBar().setValue(
                self.pulse_streamer_scroll.verticalScrollBar().maximum()
            )
        )

        self.rabi_set_btn.clicked.connect(lambda: self.set_pulse_and_count(**self._channels))
        self.rabi_set_btn.clicked.connect(self.plot_ui_init)
        self.rabi_start_btn.clicked.connect(self.rabi_start)
        self.rabi_stop_btn.clicked.connect(self.rabi_stop)
    def rabi_stop(self):

        # self.pulser.reset()
        # self.task.stop()
        self._stopConstant = True
        gc.collect()
        
    def rabi_start(self):
        self.repeat_spbx.setValue(0)
        self.pulser.reset()
        self.task.start()
        self._data_container = []
        self._stopConstant = False
        time.sleep(0.5)
        final = OutputState([self._channels['ch_aom']],0,0)
        self.pulser.stream(self.seq, -1, final)
        # Start daq in thread
        thread = Thread(
            target=self.count_data_thread_func
        )
        thread.start()

    def count_data_thread_func(self):

        n_sample = self._pulse_configuration['n_sample']
        number_of_samples = len(n_sample)*2
        inner_repeat = int(self.inner_repeat_spbx.value())
        repeat_count = int(self.repeat_spbx.value())
        while True:

            data_array = np.zeros(number_of_samples,dtype=np.uint32)
            self.reader.read_many_sample_uint32(
                data=data_array,
                number_of_samples_per_channel=number_of_samples,
                timeout=10
            )
            gate_in = data_array[0::2]
            gate_out = data_array[1::2]
            counts = gate_out - gate_in

            # contrast = signal
            self._data_container.append(counts)
            print(len(self._data_container))
            if len(self._data_container) and ((len(self._data_container) // inner_repeat) == (repeat_count+1)):
                repeat_count += 1
                self.repeat_spbx.setValue(repeat_count)
                self.odmr_data_info_msg.emit(self._data_container)
                # del data_array, gate_in, gate_out, counts
                # gc.collect()

            if self._stopConstant == True:

                self.task.stop()
                time.sleep(0.5)
                self.pulser.reset()
                break
    @property
    def configure_pulse(self):
        '''
        Global parameters
        '''
        daq_high = 100 # in ns
        '''
        T1 parameters
        '''
        t1_laser_start = int(self.t1_laser_start_spbx.value()) # in ns
        t1_laser_gate = int(self.t1_laser_gate_spbx.value())*1000 # in ns
        t1_daq_gate = int(self.t1_daq_gate_spbx.value())
        t1_daq_start = int(self.t1_daq_start_spbx.value())
        t1_span = int(self.t1_span_spbx.value())
        t1_step = int(self.t1_step_spbx.value())
        t1_inner_repeat = int(self.t1_inner_repeat_spbx.value())
        t1_n_sample = np.arange(0, t1_span+t1_step, t1_step)*1000 # in ns
        '''
        Rabi parameters
        '''
        rabi_mw_start = int(self.rabi_mw_start_spbx.value()) # in ns
        rabi_mw_gate = int(self.rabi_mw_gate_spbx.value()) # in ns
        rabi_mw_step = int(self.rabi_mw_step_spbx.value()) # in ns
        rabi_laser_start = int(self.rabi_laser_start_spbx.value()) # in ns
        rabi_laser_gate = int(self.rabi_laser_gate_spbx.value())*1000 # in ns
        
        rabi_daq_gate = int(self.rabi_daq_gate_spbx.value())
        rabi_daq_start = int(self.rabi_daq_start_spbx.value())
        rabi_inner_repeat = int(self.rabi_inner_repeat_spbx.value())
        rabi_n_sample = np.arange(0, rabi_mw_gate+rabi_mw_step, rabi_mw_step)
        '''
        Ramsey parameters
        '''
        ramsey_laser_start = int(self.ramsey_laser_start_spbx.value()) # in ns
        ramsey_laser_gate = int(self.ramsey_laser_gate_spbx.value())*1000 # in ns
        ramsey_daq_gate = int(self.ramsey_daq_gate_spbx.value())
        ramsey_daq_start = int(self.ramsey_daq_start_spbx.value())
        ramsey_tau_span = int(self.ramsey_tau_span_spbx.value())
        ramsey_tau_step = int(self.ramsey_tau_step_spbx.value())
        ramsey_mw_start = int(self.ramsey_mw_start_spbx.value())
        ramsey_mw_pi = int(self.ramsey_mw_pi_spbx.value())
        ramsey_inner_repeat = int(self.ramsey_inner_repeat_spbx.value())
        ramsey_n_sample = np.arange(0, ramsey_tau_span+ramsey_tau_step, ramsey_tau_step) # in ns
        '''
        Hahn echo parameters
        '''
        hahn_laser_start = int(self.hahn_laser_start_spbx.value()) # in ns
        hahn_laser_gate = int(self.hahn_laser_gate_spbx.value())*1000 # in ns
        hahn_daq_gate = int(self.hahn_daq_gate_spbx.value())
        hahn_daq_start = int(self.hahn_daq_start_spbx.value())
        hahn_tau_span = int(self.hahn_tau_span_spbx.value())
        hahn_tau_step = int(self.hahn_tau_step_spbx.value())
        hahn_mw_start = int(self.hahn_mw_start_spbx.value())
        hahn_mw_pi = int(self.hahn_mw_pi_spbx.value())
        hahn_inner_repeat = int(self.hahn_inner_repeat_spbx.value())
        hahn_n_sample = np.arange(0, hahn_tau_span+hahn_tau_step,hahn_tau_step) # in ns
        '''
        Optimization parameters
        '''
        opt_laser_start = int(self.opt_laser_start_spbx.value()) # in ns
        opt_laser_gate = int(self.opt_laser_gate_spbx.value())*1000 # in ns
        opt_daq_gate = int(self.opt_daq_gate_spbx.value())
        opt_daq_start = int(self.opt_daq_start_spbx.value())
        opt_tau = int(self.opt_tau_spbx.value())
        opt_pi_start = int(self.opt_pi_start_spbx.value())
        opt_pi_stop = int(self.opt_pi_stop_spbx.value())
        opt_pi_step = int(self.opt_pi_step_spbx.value())
        opt_inner_repeat = int(self.opt_inner_repeat_spbx.value())
        opt_n_sample = np.arange(opt_pi_start, opt_pi_stop+opt_pi_step,opt_pi_step) # in ns
        return{
            'T1': {
                't1_laser_start':t1_laser_start,
                't1_laser_gate':t1_laser_gate,
                't1_daq_gate':t1_daq_gate,
                't1_daq_start':t1_daq_start,
                't1_inner_repeat':t1_inner_repeat,
                't1_n_sample':t1_n_sample,
                't1_laser_part': int(t1_laser_gate/2),
                'daq_high':daq_high,
            },
            'Rabi': {
                'rabi_mw_start':rabi_mw_start,
                'rabi_mw_gate':rabi_mw_gate,
                'rabi_mw_step':rabi_mw_step,
                'rabi_laser_start':rabi_laser_start,
                'rabi_laser_gate':rabi_laser_gate,
                'daq_high':daq_high,
                'rabi_daq_gate':rabi_daq_gate,
                'rabi_daq_start':rabi_daq_start,
                'rabi_inner_repeat':rabi_inner_repeat,
                'rabi_n_sample': rabi_n_sample,
                'rabi_laser_part': int(rabi_laser_gate/2)
            },
            'Ramsey':{
                'ramsey_laser_start':ramsey_laser_start,
                'ramsey_laser_gate':ramsey_laser_gate,
                'ramsey_daq_gate':ramsey_daq_gate,
                'ramsey_daq_start':ramsey_daq_start,
                'ramsey_daq_start':ramsey_daq_start,
                'ramsey_mw_start':ramsey_mw_start,
                'ramsey_mw_pi':ramsey_mw_pi,
                'ramsey_inner_repeat':ramsey_inner_repeat,
                'ramsey_n_sample':ramsey_n_sample,
                'ramsey_laser_part': int(ramsey_laser_gate/2),
                'daq_high':daq_high,
            },
            'Hahn Echo':{
                'hahn_laser_start':hahn_laser_start,
                'hahn_laser_gate':hahn_laser_gate,
                'hahn_daq_gate':hahn_daq_gate,
                'hahn_daq_start':hahn_daq_start,
                'hahn_daq_start':hahn_daq_start,
                'hahn_mw_start':hahn_mw_start,
                'hahn_mw_pi':hahn_mw_pi,
                'hahn_inner_repeat':hahn_inner_repeat,
                'hahn_n_sample':hahn_n_sample,
                'hahn_laser_part': int(hahn_laser_gate/2),
                'daq_high':daq_high,                
            },
            'Optimization':{
                'opt_laser_start':opt_laser_start,
                'opt_laser_gate':opt_laser_gate,
                'opt_daq_gate':opt_daq_gate,
                'opt_daq_start':opt_daq_start,
                'opt_tau':opt_tau,
                'opt_inner_repeat':opt_inner_repeat,
                'opt_n_sample':opt_n_sample,
                'opt_laser_part':int(opt_laser_gate/2),
                'daq_high':daq_high
            }
        } 

    def configure_sequence(self, current_type):
        HIGH=1
        LOW=0

        seq_aom=[]
        seq_switch=[]
        seq_daq = []
        self._pulse_configuration = self.configure_pulse[current_type]      
        for key, value in self._pulse_configuration.items():
            setattr(self, key, value)  
        if current_type == 'T1':
            for item in self.t1_n_sample:
                seq_aom += [(self.t1_laser_part,HIGH),(item,LOW),(self.rabi_laser_part,HIGH)]
                seq_switch += [(self.t1_laser_gate+item,LOW)]
                seq_daq += [(self.t1_laser_part+item+self.t1_daq_start_spbx,LOW),(self.daq_high,HIGH),
                            (self.t1_daq_gate-self.daq_high,LOW),(self.daq_high,HIGH),
                            (self.t1_laser_part-self.t1_daq_start-self.daq_high-self.t1_daq_gate,LOW)]
            return seq_aom, seq_switch, seq_daq, self.t1_n_sample
        elif current_type == 'Rabi':           
            for item in self.rabi_n_sample:
                seq_aom += [(self.rabi_laser_part,HIGH),(self.rabi_mw_start+item+self.rabi_laser_start,LOW),(self.rabi_laser_part,HIGH)]
                seq_switch += [(self.rabi_laser_part+self.rabi_mw_start,LOW),(item,HIGH),(self.rabi_laser_start+self.rabi_laser_part,LOW)]
                seq_daq += [(self.rabi_laser_part+self.rabi_mw_start+item+self.rabi_daq_start,LOW),(self.daq_high,HIGH),
                            (self.rabi_daq_gate-self.daq_high,LOW),(self.daq_high,HIGH),
                            (self.rabi_laser_part-self.rabi_daq_start+self.rabi_laser_start-self.daq_high-self.rabi_daq_gate,LOW)]
            return seq_aom, seq_switch, seq_daq, self.rabi_n_sample
        elif current_type == 'Ramsey':
            for item in self.ramsey_n_sample:
                seq_aom += [(self.ramsey_laser_part,HIGH),(self.ramsey_mw_start+item+self.ramsey_laser_start+self.ramsey_mw_pi,LOW),(self.ramsey_laser_part,HIGH)]
                seq_switch += [(self.ramsey_laser_part+self.ramsey_mw_start,LOW),(int(self.ramsey_mw_pi/2),HIGH),(item,LOW),(int(self.ramsey_mw_pi/2),HIGH),
                               (self.ramsey_laser_start+self.ramsey_laser_part,LOW)]
                seq_daq += [(self.ramsey_laser_part+self.ramsey_mw_start+item+self.ramsey_daq_start+self.ramsey_mw_pi,LOW),(self.daq_high,HIGH),
                            (self.ramsey_daq_gate-self.daq_high,LOW),(self.daq_high,HIGH),
                            (self.ramsey_laser_part-self.ramsey_daq_start+self.ramsey_laser_start-self.daq_high-self.ramsey_daq_gate,LOW)]
            return seq_aom, seq_switch, seq_daq, self.ramsey_n_sample
        elif current_type == 'Hahn Echo':
            for item in self.hahn_n_sample:
                seq_aom += [(self.hahn_laser_part,HIGH),(self.hahn_mw_start+item*2+self.hahn_laser_start+self.hahn_mw_pi*2,LOW),(self.hahn_laser_part,HIGH)]
                seq_switch += [(self.hahn_laser_part+self.hahn_mw_start,LOW),(int(self.hahn_mw_pi/2),HIGH),(item,LOW),(self.hahn_mw_pi,HIGH),(item,LOW),
                               (int(self.hahn_mw_pi/2),HIGH),(self.hahn_laser_start+self.hahn_laser_part,LOW)]
                seq_daq += [(self.hahn_laser_part+self.hahn_mw_start+2*item+self.hahn_daq_start+self.hahn_mw_pi*2,LOW),(self.daq_high,HIGH),
                            (self.hahn_daq_gate-self.daq_high,LOW),(self.daq_high,HIGH),
                            (self.hahn_laser_part-self.hahn_daq_start+self.hahn_laser_start-self.daq_high-self.hahn_daq_gate,LOW)]
            return seq_aom, seq_switch, seq_daq, self.hahn_n_sample
        elif current_type == 'Optimization':
            for item in self.opt_n_sample:
                seq_aom += [(self.opt_laser_part,HIGH),(self.opt_mw_start+item*2+self.opt_laser_start+self.opt_tau*2,LOW),(self.opt_laser_part,HIGH)]
                seq_switch += [(self.opt_laser_part+self.opt_mw_start,LOW),(int(item/2),HIGH),(self.opt_tau,LOW),(item,HIGH),(self.opt_tau,LOW),
                               (int(item/2),HIGH),(self.opt_laser_start+self.opt_laser_part,LOW)]
                seq_daq += [(self.opt_laser_part+self.opt_mw_start+2*item+self.opt_daq_start+self.opt_tau*2,LOW),(self.daq_high,HIGH),
                            (self.opt_daq_gate-self.daq_high,LOW),(self.daq_high,HIGH),
                            (self.opt_laser_part-self.opt_daq_start+self.opt_laser_start-self.daq_high-self.opt_daq_gate,LOW)]
            return seq_aom, seq_switch, seq_daq, self.opt_n_sample
    def set_pulse_and_count(self, ch_aom, ch_switch, ch_daq, **kwargs):
        # print(ch_aom, ch_switch, ch_daq, ch_mw_source)
        current_tab_index = self.pulse_tab.currentIndex()
        current_tab_name = self.pulse_tab.tabText(current_tab_index)
        
        seq_aom, seq_switch, seq_daq, n_sample = self.configure_sequence(current_tab_name)

        #create the sequence
        self.seq = Sequence()
        
        #set digital channels
        self.seq.setDigital(ch_aom, seq_aom)
        self.seq.setDigital(ch_switch, seq_switch)
        self.seq.setDigital(ch_daq, seq_daq)

        self.seq.plot()
        self.task.timing.cfg_samp_clk_timing(
            rate=2E6,
            source='/Dev2/PFI1',
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=2*len(n_sample)
        )
        self.pulse_streamer_info_msg.emit('Counter input channel: '+self.odmr_ctr_channel.ci_count_edges_term)
        
        self.reader = CounterReader(self.task.in_stream)
    
    def pulser_daq_on_activate(self):
        '''
        Pusler Init
        '''
        self.pulser = self.hardware.pulser_generate()
        '''
        DAQ Init
        '''
        self.task, self.odmr_ctr_channel = self.hardware.daq_task_generate(**self._channels)
        self.pulse_streamer_info_msg.emit('DAQ Counter channel: '+self.odmr_ctr_channel.channel_names[0])
        self.pulse_streamer_info_msg.emit('DAQ APD channel: '+self.odmr_ctr_channel.ci_count_edges_term)

    def pulser_daq_on_deactivate(self):
        self.pulser.reset()
        self.task.stop()
        self.task.close()
    def pulse_streamer_info_ui(self):

        self.pulse_streamer_msg.setWordWrap(True)  # 自动换行
        self.pulse_streamer_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.pulse_streamer_msg_history = []

    def pulse_streamer_slot(self, msg):

        # print(msg)
        self.pulse_streamer_msg_history.append(msg)
        self.pulse_streamer_msg.setText("<br>".join(self.pulse_streamer_msg_history))
        self.pulse_streamer_msg.resize(700, self.pulse_streamer_msg.frameSize().height() + 20)
        self.pulse_streamer_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    '''Set window ui'''
    def window_btn_signal(self):
        # window button sigmal
        self.close_btn.clicked.connect(self.close)
        self.max_btn.clicked.connect(self.maxornorm)
        self.min_btn.clicked.connect(self.showMinimized)
        
    #create window blur
    def render_shadow(self):
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)  # 偏移
        self.shadow.setBlurRadius(30)  # 阴影半径
        self.shadow.setColor(QColor(128, 128, 255))  # 阴影颜色
        self.mainwidget.setGraphicsEffect(self.shadow)  # 将设置套用到widget窗口中

    def maxornorm(self):
        if self.isMaximized():
            self.showNormal()
            self.norm_icon = QIcon()
            self.norm_icon.addPixmap(QPixmap(":/my_icons/images/icons/max.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.norm_icon)
        else:
            self.showMaximized()
            self.max_icon = QIcon()
            self.max_icon.addPixmap(QPixmap(":/my_icons/images/icons/norm.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.max_icon)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = QPoint
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标
        
    def mouseMoveEvent(self, QMouseEvent):
        m_position = QPoint
        m_position = QMouseEvent.globalPos() - self.pos()
        width = QDesktopWidget().availableGeometry().size().width()
        height = QDesktopWidget().availableGeometry().size().height()
        if m_position.x() < width*0.7 and m_position.y() < height*0.06:
            self.m_flag = True
            if Qt.LeftButton and self.m_flag:                
                pos_x = int(self.m_Position.x())
                pos_y = int(self.m_Position.y())
                if pos_x < width*0.7 and pos_y < height*0.06:           
                    self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
                    QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    '''
    RF CONTROL
    '''
    def rf_info_ui(self):

        self.rf_msg.setWordWrap(True)  # 自动换行
        self.rf_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.rf_msg_history = []

    def rf_slot(self, msg):

        # print(msg)
        self.rf_msg_history.append(msg)
        self.rf_msg.setText("<br>".join(self.rf_msg_history))
        self.rf_msg.resize(700, self.rf_msg.frameSize().height() + 20)
        self.rf_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    def my_rf_signal(self):

        #open button signal
        self.rf_connect_btn.clicked.connect(self.boot_rf)

        #message signal
        self.rf_info_msg.connect(self.rf_slot)

        # RF scroll area scrollbar signal
        self.rf_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.rf_scroll.verticalScrollBar().setValue(
                self.rf_scroll.verticalScrollBar().maximum()
            )
        )

        # combobox restore signal
        self.rf_visa_rst_btn.clicked.connect(self.rf_cbx_test)

        # RF On button signal
        self.rf_ply_stp_btn.clicked.connect(self.rf_ply_stp)


    def rf_cbx_test(self):
        
        self.rf_cbx.clear()
        self.rm = pyvisa.ResourceManager()
        self.ls = self.rm.list_resources()
        self.rf_cbx.addItems(self.ls)

    def boot_rf(self):
        
        # Boot RF generator
        self.rf_port = self.rf_cbx.currentText()
        # print(self.rf_port)
        self._gpib_connection = self.rm.open_resource(self.rf_port)
        self._gpib_connection.write_termination = '\n'
        instrument_info = self._gpib_connection.query('*IDN?')
        
        # # 恢复出厂设置
        # self.fac = self.my_instrument.write(':SYST:PRES:TYPE FAC')
        
        # self.preset = self.my_instrument.write(':SYST:PRES')
        self._gpib_connection.write(':OUTPut:STATe OFF') # switch off the output
        self._gpib_connection.write('*RST')

        self.rf_info_msg.emit(repr(instrument_info))
        
    def rf_ply_stp(self):
        output_status = self._gpib_connection.query(':OUTPut:STATe?')
        
        if output_status == '0\n':
            frequency = float(self.cw_freq_spbx.value())*1e6
            power = float(self.cw_power_spbx.value())
            self.rf_ply_stp_btn.setText('RF OFF')
            self.off_icon = QIcon()
            self.off_icon.addPixmap(QPixmap(":/my_icons/images/icons/stop.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.off_icon)
            self._gpib_connection.write(':FREQ:MODE CW')
            self._gpib_connection.write(':FREQ:CW {0:f} Hz'.format(frequency))
            self._gpib_connection.write(':POWer:AMPLitude {0:f}'.format(power))
            rtn = self._gpib_connection.write(':OUTPut:STATe ON')
            if rtn != 0:
                self.rf_info_msg.emit('RF ON succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF ON failed')
                sys.emit()
        elif output_status == '1\n':
            self.rf_ply_stp_btn.setText('RF ON  ')
            self.on_icon = QIcon()
            self.on_icon.addPixmap(QPixmap(":/my_icons/images/icons/play.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.on_icon)
            rtn = self._gpib_connection.write(':OUTPut:STATe OFF')
            if rtn != 0:
                self.rf_info_msg.emit('RF OFF succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF OFF failed')
                sys.emit()
    def closeEvent(self, event):
        self._gpib_connection.write(':OUTPut:STATe OFF')
        self._gpib_connection.close()  
        self.rm.close()
        self.pulser_daq_on_deactivate()
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()
        return
if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec()

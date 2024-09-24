from dataclasses import dataclass
import numpy as np
import copy
import torch


@dataclass
class ProfileConfig(object):
    slots: int
    header: int
    wg_num: int
    word_per_slot: int


@dataclass
class Event(object):
    region_id: int
    start: int
    end: int
    wg: int


def get_events(index, wg_id, data):
    size = index if len(data) > index else len(data)
    event_list = []
    active_event = {}
    for i in range(0, size, 2):
        metadata = data[i]
        cycle = data[i + 1]
        is_start = metadata >> 31
        region_id = metadata & 0x7FFFFFFF
        if region_id not in active_event:
            active_event[region_id] = Event(region_id, 0, 0, wg_id)
        if is_start == 0:
            active_event[region_id].start = cycle
        else:
            active_event[region_id].end = cycle
            event_list.append(copy.deepcopy(active_event[region_id]))

    return event_list


def shift_start(event_list):
    start_time = []
    for event in event_list:
        if event.start > event.end:
            assert ("Error: cycle overflow, not support for now")
        start_time.append(event.start)
    min_start = min(start_time)
    for event in event_list:
        event.start -= min_start
        event.end -= min_start


def get_scratch_size(config):
    return config.header + config.slots * config.word_per_slot


def get_chrome_event_str(event, block_id, sm_id):
    return f'{{"name": "region_{event.region_id}", "cat": "triton", "ph": "X", "ts": {event.start}, "dur": {event.end - event.start}, "pid": "{block_id}", "tid": "{event.wg}", "args":{{"sm_id": "{sm_id}", "frequency": "1MHz"}}}}'


def dump_chrome_trace(block_num, config, profile_mem, file_name):
    scratch = get_scratch_size(config)
    trace_str = "{\"traceEvents\": ["
    for i in range(block_num):
        workspace = profile_mem[i * scratch:(i + 1) * scratch]
        block_id = workspace[0].item()
        sm_id = workspace[1].item()
        index = workspace[2].item()
        data = workspace[3:].tolist()
        event_list = []
        wg_data_len = int(len(data) / config.wg_num)
        for j in range(config.wg_num):
            ws_index = j * wg_data_len
            event_list += get_events(index, j, data[ws_index:ws_index + wg_data_len])

        shift_start(event_list)
        for event in event_list:
            chrome_event_str = get_chrome_event_str(event, block_id, sm_id)
            trace_str += chrome_event_str + ",\n"

    trace_str = trace_str[:-2] + "]}"

    with open(file_name, "w") as f:
        f.write(trace_str)

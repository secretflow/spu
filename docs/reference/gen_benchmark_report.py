#! /usr/bin/python3

# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import pandas as pd
import numpy as np
import os
from enum import Enum
import argparse

g_time_list = [
    ('ns', 1000),
    ('us', 1000),
    ('ms', 1000),
    ('s', 60),
    ('min', 60),
    ('h', 60),
]


def formal_time(count: float, unit: str):
    is_unit_good = False
    for i in range(len(g_time_list) - 1):
        if g_time_list[i][0] == unit:
            if count > g_time_list[i][1]:
                unit = g_time_list[i + 1][0]
                count /= g_time_list[i][1]
            else:
                break
    return count, unit


def get_time_unit(count: float, unit: str):
    _, unit = formal_time(count, unit)
    return unit


def trans_rep(counts):
    if counts == 2**10 or counts == 1000 or counts == '1000' or counts == '1024':
        return '1k'
    if counts == 2**20 or counts == 1e6 or counts == '1000000' or counts == '1048576':
        return '1M'
    return str(counts)


class FileType(Enum):
    XLS = 1
    HTML = 2
    MD = 3
    CSV = 4


def list_join(df, cols, sp):
    ret = []
    for i in range(df.shape[0]):
        cur = []
        for c in cols:
            cur.append(df[c][i])
        ret.append(sp.join(cur))
    return ret


def time_exchange(time_val, o_unit, n_unit):
    if o_unit == n_unit:
        return time_val

    def get_index(unit):
        for i in range(len(g_time_list)):
            if g_time_list[i][0] == unit:
                return i
        raise ValueError(f'{unit} is not support')

    o_index = get_index(o_unit)
    n_index = get_index(n_unit)
    if o_index < n_index:
        for i in range(o_index, n_index):
            time_val /= g_time_list[i][1]
    else:
        for i in range(n_index, o_index):
            time_val *= g_time_list[i][1]
    return time_val


class TableFromat:
    def __init__(self, rows: list, cols: list, values: list, pick_rows: list):
        self.rows = rows
        self.cols = cols
        self.values = values
        self.pick_row = pick_rows

    def real_time_weight(self, df: pd.DataFrame, unit: str):
        a = []
        for row_id in df.index:
            a.append(
                time_exchange(
                    df.loc[row_id, 'real_time'], df.loc[row_id, 'time_unit'], unit
                )
            )
        return (np.array(a) / 100).mean()

    def format_time(self, df: pd.DataFrame, unit: str):
        real_time = df['real_time']
        time_unit = df['time_unit']
        for i in real_time.index:
            df.loc[i, 'time'] = '{:>10.2f}'.format(
                time_exchange(real_time[i], time_unit[i], unit)
            )
        return df

    def uniform_time(self, df: pd.DataFrame, unit: str):
        df['time'] = pd.NA
        df = self.format_time(df, unit)
        df[self.cols[-1]] += '/' + unit
        return df

    def _extra_work(self, df: pd.DataFrame):
        return self.uniform_time(df, "ms")

    def format_time_with_last_col(self, df: pd.DataFrame):
        df['time'] = pd.NA
        time_col_name = self.cols[-1]
        time_col_values = [i for i in df[time_col_name].unique().tolist() if i]
        for time_col_value in time_col_values:
            weight = self.real_time_weight(
                df[df[time_col_name] == time_col_value], 'ns'
            )
            unit = get_time_unit(weight, 'ns')
            formated = self.format_time(df[df[time_col_name] == time_col_value], unit)
            for i in formated.index:
                df.loc[i, :] = formated.loc[i, :]
            df[time_col_name].replace(
                time_col_value,
                time_col_value + '/' + unit,
                inplace=True,
            )
        return df

    def reorder(self, df: pd.DataFrame):
        if not self.pick_row:
            return df
        cur_index = set()
        if df.index.nlevels == 1:
            cur_index = set([i for i in df.index])
        else:
            cur_index = set([i for i in df.index.get_level_values(0)])
        reorder_index = []
        for p in self.pick_row:
            if p in cur_index:
                reorder_index.append(p)
                cur_index.discard(p)
        reorder_index.extend([i for i in cur_index])
        return df.loc[reorder_index, :]

    def format(self, df: pd.DataFrame):
        for r in self.rows + self.cols:
            df = df[df[r].notna()]
            df = df[df[r] != '-']
        df = self._extra_work(df)
        resv = self.rows + self.cols + self.values
        drops = [col for col in df.columns if col not in resv]
        df.drop(columns=drops, inplace=True)
        return self.reorder(
            df.pivot(index=self.rows, columns=self.cols, values=self.values)
        )


class BenchmarkTable:
    def __init__(self, extra: dict = {}, benchmarks: list = []):
        self.heads = []
        self.entrys = []
        self.df = pd.DataFrame()
        self.extra = extra
        self._parse(benchmarks)

    def _parse(self, benchmarks: list):
        for entry in benchmarks:
            self._parse_entry(entry)
        self.df = self._gen_dataframe()

    def _gen_dataframe(self):
        cols = set()
        for head in self.heads:
            for col in head:
                cols.add(col)
        cols = [_ for _ in cols]
        cont = []
        for i in range(len(self.heads)):
            row = [pd.NA] * len(cols)
            for n in range(len(self.heads[i])):
                row[cols.index(self.heads[i][n])] = self.entrys[i][n]
            cont.append(row)

        ret = pd.DataFrame(cont, columns=cols)
        for k, v in self.extra.items():
            ret[k] = str(v)
        return ret

    def _parse_entry(self, entry: dict):
        label = entry.get('label', '')
        ks, vs = [], []
        for k, v in entry.items():
            ks.append(k)
            vs.append(v)
        infos = [item.split(':') for item in label.split('/')]
        self.heads.append([info[0] for info in infos] + ks)
        self.entrys.append([info[1] for info in infos] + vs)

    def join(self, other):
        self.df = pd.concat([self.df, other.df])
        self.df.reset_index(inplace=True, drop=True)

    def sep_by(self, col_name: str):
        ret = {}
        if col_name not in self.df.columns:
            print('not')
            return ret
        sheets_name = self.df[col_name].unique()
        for name in sheets_name:
            tmp = BenchmarkTable()
            tmp.df = self.df[self.df[col_name] == name]
            ret[name] = tmp
        return ret

    def apply_format(self, format):
        return format.format(self.df)


class BenchmarkManager:
    def __init__(self):
        self.table = BenchmarkTable()
        self.sheets = {}
        self.col_time_unit = {}

    def add_report(self, filename: str):
        with open(filename, 'r') as ifile:
            report = json.load(ifile)
            extra = report['context']
            extra['env'] = os.path.basename(filename).split('.')[0]
            self.table.join(BenchmarkTable(extra, report["benchmarks"]))

    def sep_sheet(self, col_name: str):
        self.sheets.update(self.table.sep_by(col_name))

    def _dump_excel(self, format, ofilename):
        with pd.ExcelWriter(ofilename, engine='xlsxwriter') as writer:
            for sheet, table in self.sheets.items():
                print(f'{sheet}')
                table.apply_format(format).to_excel(writer, sheet_name=sheet)

    def dump(self, format, ofilename):
        if len(self.sheets) == 0:
            self.sheets['benchmark'] = self.table
        if ofilename.endswith('.xlsx'):
            self._dump_excel(format, ofilename)
            return
        func = None
        if ofilename.endswith('.md'):
            func = lambda ofile, table: table.apply_format(format).to_markdown(ofile)
        elif ofilename.endswith('.csv'):
            func = lambda ofile, table: table.apply_format(format).to_csv(ofile)
        elif ofilename.endswith('.html'):
            func = lambda ofile, table: table.apply_format(format).to_html(ofile)
        else:
            raise NotImplemented(f"{ofilename} is not support")
        with open(ofilename, 'w') as ofile:
            for sheet, table in self.sheets.items():
                print(f'\n{sheet}\n', file=ofile)
                func(ofile, table)
        print(f'output: {ofilename}')


class ReportGen:
    def __init__(self):
        self.manager = BenchmarkManager()

    def parse_report(self, filename: str):
        self.manager.add_report(filename)

    def sep_sheet(self, sheet_col):
        self.manager.sep_sheet(sheet_col)

    def dump(self, format, ofilename):
        self.manager.dump(format, ofilename)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        description='format and export benchmark result',
        epilog='''eg: --output=test.xlsx
            --input=LAN.json,WAN.json
            --columns=env,buf_len
            --rows=op_name,field_type
            --values=time
            --sheet="Benchmark Protocol"''',
    )
    parse.add_argument(
        '--input',
        nargs=1,
        help='filename will be used to indicate "env", eg: --input=LAN.json,WAN.json',
    )
    parse.add_argument(
        '--output',
        nargs=1,
        help='output filename, suffix should be in [md, xlsx, csv, html]',
    )
    parse.add_argument(
        '--columns', nargs=1, help='column labels, eg: --columns=label1,label2'
    )
    parse.add_argument('--rows', nargs=1, help='row labels, eg: --rows=label1,label2')
    parse.add_argument('--pick_rows', nargs=1, help='values of row label1')
    parse.add_argument(
        '--values', nargs=1, help='value labels, eg: --values=label1,label2'
    )
    parse.add_argument('--sheet', nargs=1, help='sheets label, should just be one')

    args = parse.parse_args(
        # [
        #     '--output=report.md',
        #     '--input=../LAN.json,../WAN_300mbit_20msec.json',
        #     '--columns=env,buf_len',
        #     '--rows=op_name,field_type',
        #     '--values=time',
        #     '--sheet=Benchmark Protocol',
        #     '--pick_rows=xor_ss,add_ss',
        # ]
    )
    gen = ReportGen()
    for infile in args.input[0].split(','):
        print(infile)
        gen.parse_report(infile)
    gen.sep_sheet(args.sheet[0])
    format = TableFromat(
        args.rows[0].split(','),
        args.columns[0].split(','),
        args.values[0].split(','),
        args.pick_rows[0].split(','),
    )
    gen.dump(format, args.output[0])

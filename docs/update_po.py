import os
import re

ROOT = './locales/zh_CN/LC_MESSAGES'
PO_FILES = list(filter(lambda x: x[-3:] == '.po' and x != "index.po", os.listdir(ROOT)))

for po_file in PO_FILES:
    src_ctx = open(f"{ROOT}/{po_file}", 'r').read()

    for sub_po_file in  os.listdir(f"{ROOT}/{po_file[:-3]}"):
        dst_ctx = open(f"{ROOT}/{po_file[:-3]}/{sub_po_file}", 'r').read()

        for title in re.findall('#: (.*?)\n', dst_ctx):
            src_title = src_ctx.find(title)
            if src_title < 0:
                continue

            dst_title = dst_ctx.find(title)
            if dst_title < 0:
                continue

            src_start = src_ctx.find('msgstr', src_title)
            src_end = src_ctx.find('\n', src_start)

            if src_end - src_start <= len('msgstr ""'):
                continue

            dst_start = dst_ctx.find('msgstr', dst_title)
            dst_end = dst_ctx.find('\n', dst_start)

            dst_ctx = dst_ctx[:dst_start] + src_ctx[src_start:src_end] + dst_ctx[dst_end:] 

        
        open(f"{ROOT}/{po_file[:-3]}/{sub_po_file}", 'w').write(dst_ctx)

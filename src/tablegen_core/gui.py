"""Small bilingual GUI. Generation runs in a cancellable subprocess."""
from datetime import datetime
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import webbrowser


from .gui_text import TEXT_EN, PRIOR_HELP, MANUAL, MODE_LABELS, POOL_LABELS


class CoreApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from .registry import selectable_priors
        self.root = root
        self.process = None
        self.files = None
        self.job = None
        self.log = None
        self.language = tk.StringVar(value='한국어')
        self._translated_widgets = []
        self._status_key = '처음에는 기본값으로 시작하세요. 기존 폴더는 덮어쓰지 않습니다.'
        self._status_args = {}
        root.geometry('980x740')
        root.protocol('WM_DELETE_WINDOW', self.close)
        top = ttk.Frame(root, padding=(18, 8))
        top.pack(fill='x')
        ttk.Label(top, text='Language / 언어').pack(side='right')
        self.language_control = ttk.Combobox(top, textvariable=self.language, values=['한국어', 'English'], state='readonly', width=12)
        self.language_control.pack(side='right', padx=8)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        panel = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(panel, text='표 생성')
        help_panel = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(help_panel, text='사용 설명서')
        self.manual = tk.Text(help_panel, wrap='word', padx=14, pady=12, font=('', 13))
        scrollbar = ttk.Scrollbar(help_panel, command=self.manual.yview)
        self.manual.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.manual.pack(fill='both', expand=True)
        self.label(panel, text='설정을 고르고 표를 만드세요', font=('', 19)).pack(anchor='w')
        self.label(panel, text='전체 24종만 포함한 생성기입니다. 표는 내 컴퓨터에서 계산하고 저장합니다.').pack(anchor='w', pady=6)
        form = ttk.Frame(panel)
        form.pack(fill='x')
        self.values = {}
        fields = [('rows','행 수','128','표에 들어갈 사례 수'),('features','입력 열 수','4','사례를 설명하는 항목 수'),('targets','정답 열 수','1','예측할 결과 항목 수'),('seed','시드','42','같은 설정으로 같은 표를 다시 만드는 번호'),('min_n_nodes','최소 노드','4','DAG 내부 계산 지점 수의 하한'),('max_n_nodes','최대 노드','12','DAG 내부 계산 지점 수의 상한'),('cauchy_dag_offset','연결 성향','0','DAG에서 큰 값일수록 평균적으로 연결이 많아짐'),('classes','정답 범주 수','0','0은 수치형, 2 이상은 분류용 범주 수 (DAG)')]
        for i,(key,label,value,hint) in enumerate(fields):
            self.values[key] = tk.StringVar(value=value)
            self.label(form,text=label).grid(row=i,column=0,sticky='w',pady=3)
            ttk.Entry(form,textvariable=self.values[key],width=14).grid(row=i,column=1,sticky='w')
            self.label(form,text=hint).grid(row=i,column=2,sticky='w',padx=12)
        self.mode = tk.StringVar(value='연결 방식 (DAG)')
        self.prior = tk.StringVar(value='기본 8종')
        self.label(form,text='생성 방식').grid(row=8,column=0,sticky='w')
        self.mode_box = ttk.Combobox(form,textvariable=self.mode,values=MODE_LABELS['ko'],state='readonly')
        self.mode_box.grid(row=8,column=1,columnspan=2,sticky='ew')
        self.label(form,text='생성 규칙').grid(row=9,column=0,sticky='w')
        self.prior_box = ttk.Combobox(form,textvariable=self.prior,values=POOL_LABELS['ko']+selectable_priors(),state='readonly',width=50)
        self.prior_box.grid(row=9,column=1,columnspan=2,sticky='ew')
        self.prior_help = tk.StringVar()
        self.prior.trace_add('write', self.explain)
        self.explain()
        ttk.Label(panel, textvariable=self.prior_help, wraplength=920).pack(anchor='w', pady=4)
        self.output = tk.StringVar(value=str(Path.cwd()/'output'/datetime.now().strftime('%Y%m%d_%H%M%S')))
        self.label(form,text='새 결과 폴더').grid(row=10,column=0,sticky='w')
        ttk.Entry(form,textvariable=self.output,width=55).grid(row=10,column=1,columnspan=2,sticky='ew')
        self.button(form,text='저장 위치 선택',command=self.browse).grid(row=10,column=3,padx=8)
        self.status = tk.StringVar(value='처음에는 기본값으로 시작하세요. 기존 폴더는 덮어쓰지 않습니다.')
        ttk.Label(panel,textvariable=self.status,wraplength=920).pack(anchor='w',pady=10)
        actions = ttk.Frame(panel);actions.pack(fill='x')
        self.generate_button = self.button(actions,text='표 생성하고 저장하기',command=self.start);self.generate_button.pack(side='left')
        self.cancel_button = self.button(actions,text='취소',command=self.cancel,state='disabled');self.cancel_button.pack(side='left',padx=8)
        self.open_button = self.button(actions,text='시각화 열기 (3D)',command=self.open_report,state='disabled');self.open_button.pack(side='left')
        self.table = ttk.Treeview(panel,show='headings',height=8);self.table.pack(fill='both',expand=True,pady=10)
        self.label(panel,text='화면은 앞 20행입니다. 전체 표는 CSV·NPY, 설정은 JSON, 회전 가능한 3D 그림은 HTML에 저장합니다.').pack(anchor='w')
        self.language.trace_add('write', self.switch_language)
        self.switch_language()
        root.after(100,self.poll)

    @property
    def lang(self):
        return 'en' if self.language.get() == 'English' else 'ko'

    def tr(self, text):
        return TEXT_EN.get(text, text) if self.lang == 'en' else text

    def label(self, parent, **kwargs):
        from tkinter import ttk
        return self.translated_widget(ttk.Label, parent, **kwargs)

    def button(self, parent, **kwargs):
        from tkinter import ttk
        return self.translated_widget(ttk.Button, parent, **kwargs)

    def translated_widget(self, constructor, parent, **kwargs):
        original = kwargs.get('text')
        if original is not None: kwargs['text'] = self.tr(original)
        widget = constructor(parent, **kwargs)
        if original is not None: self._translated_widgets.append((widget, original))
        return widget

    def mode_key(self):
        for labels in MODE_LABELS.values():
            if self.mode.get() in labels: return ('dag', 'direct')[labels.index(self.mode.get())]
        raise ValueError('Unknown generation mode')

    def prior_key(self):
        for labels in POOL_LABELS.values():
            if self.prior.get() in labels: return ('default8', 'all24', 'robust16')[labels.index(self.prior.get())]
        return self.prior.get()

    def explain(self, *_):
        self.prior_help.set(PRIOR_HELP[self.lang].get(self.prior_key(), self.tr('생성 규칙을 선택하세요.')))

    def set_status(self, key, **kwargs):
        self._status_key, self._status_args = key, kwargs
        self.status.set(self.tr(key).format(**kwargs))

    def switch_language(self, *_):
        from ._version import __version__
        from .registry import selectable_priors
        mode, prior = self.mode_key(), self.prior_key()
        self.root.title(f'Tablegen Core {__version__} — ' + self.tr('24종 합성 표 생성기'))
        for widget, original in self._translated_widgets: widget.configure(text=self.tr(original))
        self.mode_box.configure(values=MODE_LABELS[self.lang])
        self.mode.set(MODE_LABELS[self.lang][('dag', 'direct').index(mode)])
        self.prior_box.configure(values=POOL_LABELS[self.lang] + selectable_priors())
        self.prior.set(POOL_LABELS[self.lang][('default8', 'all24', 'robust16').index(prior)] if prior in ('default8', 'all24', 'robust16') else prior)
        self.notebook.tab(0, text=self.tr('표 생성'))
        self.notebook.tab(1, text=self.tr('사용 설명서'))
        self.manual.configure(state='normal')
        self.manual.delete('1.0', 'end')
        self.manual.insert('1.0', MANUAL[self.lang].format(version=__version__))
        self.manual.configure(state='disabled')
        self.set_status(self._status_key, **self._status_args)
        self.explain()

    def browse(self):
        from tkinter import filedialog
        parent = filedialog.askdirectory(parent=self.root,title=self.tr('결과를 보관할 상위 폴더'))
        if parent:
            self.output.set(str(Path(parent)/datetime.now().strftime('table_%Y%m%d_%H%M%S_%f')))

    def start(self):
        from tkinter import messagebox
        if self.process is not None:return
        try:
            from .api import GenerationConfig
            cfg = {key:int(self.values[key].get()) for key in ('rows','features','targets','seed')}
            cfg['mode'] = self.mode_key()
            cfg['priors'] = [self.prior_key()]
            classes = int(self.values['classes'].get())
            if classes < 0 or classes == 1:raise ValueError(self.tr('정답 범주 수는 0 또는 2 이상이어야 합니다.'))
            if cfg['mode'] == 'dag':
                cfg['graph_options'] = {k:int(self.values[k].get()) for k in ('min_n_nodes','max_n_nodes')}
                cfg['graph_options']['cauchy_dag_offset'] = float(self.values['cauchy_dag_offset'].get())
                if classes:cfg['target_categories'] = [classes]*cfg['targets']
            elif classes:raise ValueError(self.tr('범주형 정답은 연결 방식 (DAG)을 선택하세요.'))
            if cfg['mode']=='direct' and cfg['priors'][0] in ('default8','all24','robust16'):
                raise ValueError(self.tr('직접 방식에서는 lin 같은 생성 규칙 하나를 선택하세요.'))
            GenerationConfig.from_dict(cfg).validated()
            if not self.output.get().strip():raise ValueError(self.tr('새 결과 폴더를 지정하세요.'))
            directory = Path(self.output.get()).expanduser().absolute()
            if directory.exists():raise ValueError(self.tr('이미 있는 폴더입니다. 새 폴더 이름을 지정하세요.'))
            self.job = tempfile.TemporaryDirectory(prefix='tablegen-core-gui-')
            config_path = Path(self.job.name)/'config.json';config_path.write_text(json.dumps(cfg))
            self.log = open(Path(self.job.name)/'worker.log','w+')
            self.process = subprocess.Popen([sys.executable,'-m','tablegen_core','generate','--config',str(config_path),'--output',str(Path(self.job.name)/'result')],stdout=self.log,stderr=subprocess.STDOUT)
            self.pending_dir = directory
            self.generate_button.configure(state='disabled');self.cancel_button.configure(state='normal')
            self.set_status('생성 중입니다. 완료하면 지정한 폴더에 자동 저장합니다.')
        except Exception as exc:
            self.cleanup()
            messagebox.showerror(self.tr('설정을 확인하세요'),str(exc),parent=self.root)

    def cleanup(self):
        if self.log:self.log.close();self.log=None
        if self.job:self.job.cleanup();self.job=None

    def poll(self):
        if self.process is not None and self.process.poll() is not None:
            from tkinter import messagebox
            code=self.process.returncode;self.process=None
            self.generate_button.configure(state='normal');self.cancel_button.configure(state='disabled')
            try:
                if code:
                    self.log.flush();self.log.seek(0)
                    raise RuntimeError(self.log.read()[-3000:])
                # The worker writes only inside our temporary job. Publish on the
                # Tk event thread after successful exit, so Cancel cannot interrupt
                # copying into the user's destination.
                from .outputs import publish_staged
                publish_staged(Path(self.job.name)/'result', self.pending_dir)
                import numpy as np
                data=np.load(self.pending_dir/'table.npy',allow_pickle=False)
                meta=json.loads((self.pending_dir/'metadata.json').read_text())
                self.table.delete(*self.table.get_children());self.table.configure(columns=meta['columns'])
                for column in meta['columns']:
                    self.table.heading(column,text=column);self.table.column(column,width=100)
                for row in data[:20]:self.table.insert('','end',values=[f'{v:.6g}' for v in row])
                self.files={n:str(self.pending_dir/n) for n in ('table.csv','table.npy','metadata.json','report.html')}
                self.open_button.configure(state='normal')
                self.set_status('{rows}행 저장 완료: {path}', rows=len(data), path=self.pending_dir)
            except Exception as exc:
                self.set_status('생성하지 못했습니다. 설정과 저장 위치를 확인하세요.')
                messagebox.showerror(self.tr('생성 오류'),str(exc),parent=self.root)
            finally:self.cleanup()
        self.root.after(100,self.poll)

    def cancel(self):
        if self.process is not None:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process=None;self.cleanup()
            self.generate_button.configure(state='normal');self.cancel_button.configure(state='disabled')
            self.set_status('취소했습니다. 완료된 결과가 있는지는 지정한 폴더에서 확인하세요.')

    def open_report(self):
        if self.files:webbrowser.open(Path(self.files['report.html']).resolve().as_uri())

    def close(self):
        from tkinter import messagebox
        if self.process is not None:
            if not messagebox.askyesno(self.tr('작업 중'),self.tr('생성을 취소하고 닫을까요?'),parent=self.root):return
            self.cancel()
        self.root.destroy()


def main():
    import tkinter as tk
    root=tk.Tk();CoreApp(root);root.mainloop()


if __name__=='__main__':main()

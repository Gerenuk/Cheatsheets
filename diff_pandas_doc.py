commands=[]

for html_filename in glob.glob("*.html"):
    try:
        html=pd.read_html(html_filename)
    except Exception as exc:
        print(f"ERROR {html_filename}: {exc}")
        
    for table in html:
        for val in table.values[:,0]:
            if pd.isna(val):
                continue
            m=re.match(r"([A-Za-z0-9_]+\.)([A-Za-z0-9_]+)", val)
            if not m:
                print("?", val)
                continue
            commands.append((m.group(1), m.group(2)))
pipe(commands, groupby(get(0)), valmap(pluck(1)), valmap(set))
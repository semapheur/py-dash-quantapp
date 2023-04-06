from dash import dcc, html

label_style = 'relative flex justify-between items-center w-12 h-6 px-0.5 rounded-full bg-secondary/50 border-text/50 border hover:shadow cursor-pointer text-sm before:content-["☀"] after:content-["🌙"]'

toggle_style = 'absolute left-0.5 top-0.5 w-5 h-5 grid place-items-center rounded-full bg-secondary text-xs peer-checked:translate-x-[1.375rem] transition-transform '
toggle = html.Span(className=toggle_style)

def ThemeToggle():
  return dcc.Checklist(
    id='theme-toggle', 
    className='h-full flex items-center',
    inputClassName='peer hidden',
    labelClassName=label_style, 
    options=[
      {'label': toggle, 'value': 'dark'}
    ]
  )
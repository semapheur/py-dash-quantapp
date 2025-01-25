from dash import dcc, html

label_style = (
  "relative justify-between items-center "
  "w-10 h-6 px-0.5 rounded-full bg-secondary/50 border-text/50 border "
  "hover:shadow-sm cursor-pointer text-sm before:content-['☀'] after:content-['🌙']"
)
toggle_style = (
  "absolute grid place-items-center "
  "left-0.5 top-0.5 w-5 h-5 rounded-full bg-secondary text-xs "
  "peer-checked:translate-x-[0.8rem] transition-transform"
)


toggle = html.Span(className=toggle_style)


def ThemeToggle():
  return dcc.Checklist(
    id="theme-toggle",
    className="h-full flex items-center",
    inputClassName="peer hidden",
    labelClassName=label_style,
    labelStyle={"display": "flex"},
    options=[{"label": toggle, "value": "dark"}],
  )

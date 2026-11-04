"""Small calendar for selecting only dates present in saved verification data."""
from __future__ import annotations

import calendar
from datetime import date

import gradio as gr

JS = """
element.addEventListener('toggle', event => {
  if (event.target.tagName === 'DETAILS') element.style.zIndex = event.target.open ? '100' : '';
}, true);
element.addEventListener('click', event => {
  const button = event.target.closest('button[data-date], button[data-month]');
  if (button && !button.disabled) {
    trigger('input', {date: button.dataset.date, month: button.dataset.month});
  }
});
"""
CSS = """
position: relative; overflow: visible;
summary { cursor: pointer; border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 10px; }
.calendar { position: absolute; z-index: 50; width: 280px; background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 10px; box-shadow: 0 4px 12px #0002; }
nav { display: flex; align-items: center; justify-content: space-between; }
table { width: 100%; table-layout: fixed; text-align: center; border-collapse: collapse; }
th { font-size: 11px; padding: 4px 0; }
button { min-width: 28px; min-height: 28px; border-radius: 4px; cursor: pointer; }
button:disabled { opacity: .3; cursor: default; }
button[aria-pressed="true"] { background: var(--button-primary-background-fill); color: var(--button-primary-text-color); }
button:focus-visible, summary:focus-visible { outline: 2px solid var(--color-accent); }
"""


def selected(value):
    return value.get("date") if isinstance(value, dict) else value


def value(days, chosen=None, month=None, opened=False):
    days = sorted(set(days))
    chosen = selected(chosen)
    if chosen not in days:
        chosen = days[0] if days else None
    months = sorted({day[:7] for day in days})
    month = month if month in months else chosen[:7] if chosen else None
    if month is None:
        return {"date": None, "month": None, "html": '<span>No verified dates available</span>'}
    year, number = map(int, month.split("-"))
    position = months.index(month)
    def navigation(offset, label):
        target = position + offset
        return (f'<button type="button" data-month="{months[target]}" aria-label="{label}">{"‹" if offset < 0 else "›"}</button>'
                if 0 <= target < len(months) else f'<button type="button" disabled aria-label="{label}">{"‹" if offset < 0 else "›"}</button>')
    rows = []
    for week in calendar.monthcalendar(year, number):
        cells = []
        for day in week:
            if not day:
                cells.append("<td></td>")
                continue
            iso = date(year, number, day).isoformat()
            enabled = iso in days
            cells.append(f'<td><button type="button" data-date="{iso}" aria-label="{iso}" '
                         f'aria-pressed="{str(iso == chosen).lower()}" {"" if enabled else "disabled"}>{day}</button></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    heading = date(year, number, 1).strftime("%B %Y")
    html = (f'<details {"open" if opened else ""}><summary>{chosen} ▾</summary><div class="calendar">'
            f'<nav>{navigation(-1, "Previous available month")}<strong>{heading}</strong>'
            f'{navigation(1, "Next available month")}</nav><table aria-label="Available verification dates">'
            '<thead><tr>' + ''.join(f'<th scope="col">{day}</th>' for day in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></details>')
    return {"date": chosen, "month": month, "html": html}


def build(days, chosen, label):
    return gr.HTML(value(days, chosen), label=label, show_label=True, html_template="${value.html}",
                   css_template=CSS, js_on_load=JS, container=True, padding=True)

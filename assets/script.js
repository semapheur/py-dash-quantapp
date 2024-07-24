/**
 * @param {HTMLElement} el 
 * @param {boolean} withChildren 
 */
function removeAllEventListeners(el, withChildren = false) {
  if (withChildren) {
    el.parentNode.replaceChild(el.cloneNode(true), el)
  }
  else {
    const newEl = el.cloneNode(false)
    while (el.hasChildNodes()) newEl.appendChild(el.firstChild)
    el.parentNode.replaceChild(newEl, el)
  }
}

/**
 * @param {string} dialogId 
 */
function openDialog(dialogId) {
  dialogId = checkId(dialogId)
  const dialog = document.getElementById(dialogId)
  dialog.showModal()

  dialog.addEventListener("click", e => {
    const dialogDimensions = dialog.getBoundingClientRect()
    if (
      e.clientX < dialogDimensions.left ||
      e.clientX > dialogDimensions.right ||
      e.clientY < dialogDimensions.top ||
      e.clientY > dialogDimensions.bottom
    ) {
      dialog.close()
      removeAllEventListeners(dialog, false)
    }
  })
}

/**
 * @param {string} dialogId 
 */
function closeDialog(dialogId) {
  dialogId = checkId(dialogId)
  const dialog = document.getElementById(dialogId)
  dialog.close()
  removeAllEventListeners(dialog, false)
}

/**
 * @param {string|Object} id
 * @returns {string}
 */
function checkId(id) {
  if (typeof id !== "object") { return id }

  sorted_keys = Object.keys(id).sort()
  const sorted_object = {}
  sorted_keys.forEach((key) => {
    sorted_object[key] = id[key]
  })
  return JSON.stringify(sorted_object)
}

/**
 * @param {string} str
 * @return {boolean}
 */
function isNumber(str) {
  return !Number.isNaN(Number(str))
}

function checkKeys(obj, patterns) {
  const keys = Object.keys(obj)
  return patterns.some(pattern => keys.some(key => key.includes(pattern)))
}

/**
 * @param {Object} data
 * @param {string} startDate
 * @param {string} endDate
 * @return {Object}
 */
function sliceTimeSeries(data, startDate, endDate) {
  const start = new Date(startDate)
  const end = new Date(endDate)

  const indices = data.x.reduce((acc, dateStr, index) => {
    const date = new Date(dateStr)
    if (date >= start && date <= end) {
      acc.push(index)
    }
    return acc
  }, [])

  return {
    x: indices.map(i => data.x[i]),
    y: indices.map(i => data.y[i])
  }
}

function quoteGraphRange(figure, startDate, endDate) {
  for (const key in figure.layout) {
    if (key.startsWith("xaxis")) {
      figure["layout"][key]["range"] = [startDate, endDate]
    } 
    else if (key.startsWith("yaxis")) {
      let axisLabel = key.replace("axis", "")
      
      const yMin = []
      const yMax = []
      for (const trace of figure.data) {
        if (trace.yaxis !== axisLabel) { continue }

        let data = {x: trace.x, y: trace.y}
        data = sliceTimeSeries(data, startDate, endDate)

        yMin.push(Math.min(...data.y))
        yMax.push(Math.max(...data.y))

      }
      figure["layout"][key]["range"] = [Math.min(...yMin), Math.max(...yMax)]
      figure["layout"][key]["autorange"] = false
    }
  }
  return figure
}

/**
 * @param {Object} figure
 * @param {"line" | "candlestick"} plotType
 * @return {Object}
 */
function updateQuoteGraphType(figure, plotType) {
  const newFigure = {...figure}

  if (plotType === 'line') {
    if (figure.data[0].type === 'scatter') { return figure }

    const {x, open, high, low, close} = figure.data[0]
    const customdata = x.map((_, index) => {
      [open[index], high[index], low[index]]
    })

    newFigure.data[0] = {
      type: 'scatter',
      mode: 'lines',
      x: x,
      y: close,
      customdata: customdata
    }
  } else if (plotType === "candlestick") {
    if (figure.data[0].type === 'candlestick') { return figure }

    const {x, y, customdata} = figure.data[0]

    newFigure.data[0] = {
        type: 'candlestick',
        x: x,
        open: customdata.map(d => d[0]),
        high: customdata.map(d => d[1]),
        low: customdata.map(d => d[2]),
        close: y,
    }
  } 
  return newFigure
}

/**
 * @param {Object} relayoutData
 * @param {Object} figure
 * @return {Object}
 */
function updateGraphRelayout(relayoutData, figure) {
  function getAxes(obj) {
    const pattern = /^(x|y)axis\d?/
    const axes = new Set()
  
    for (const key in obj) {
      const match = key.match(pattern)
      if (match) {
        axes.add(match[0])
      }
    }
    return Array.from(axes)
  }
  
  const triggered_id = dash_clientside.callback_context.triggered_id
  console.log(triggered_id)

  const axes = getAxes(figure.layout)
  if (checkKeys(relayoutData, ["range[0]", "range[1]"])) {
    figure = quoteGraphRange(figure, relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"])
  }
  else if (checkKeys(relayoutData, ["autorange", "showspikes"])) {
    for (const axis of axes) {
      figure.layout[axis].autorange = true
    }
  }
  return figure
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    /**
     * @param {Object} cell
     * @param {string|Object} dialogId
     * @returns {Object}
     */
    dcf_factor_modal: function(cell, dialogId) {
      if (cell === undefined || cell.colId != "factor") { return cell }

      openModal(dialogId)
      return cell
    },
    /**
     * @param {Object} cell
     * @param {string} dialogId
     * @returns {string}
     */
    cellClickModal: function(table_cell, dialogId) {
      if (table_cell !== undefined) { openDialog(dialogId) }
      
      return window.dash_clientside.no_update
    },
    /**
     * @param {number} nClicks
     * @param {Object|string} dialogId
     * @returns {string}
     */
    openModal: function(nClicks, dialogId) {
      if (nClicks > 0) { openDialog(dialogId) }
        
      return window.dash_clientside.no_update
    },
    /**
     * @param {number} nClicks
     * @param {string} dialogId
     * @returns {string}
     */
    closeModal: function(nClicks, dialogId) {
      if (nClicks > 0) {closeDialog(dialogId)}

      return window.dash_clientside.no_update
    },
    /**
     * @param {number} openStamp
     * @param {number} closeStamp
     * @param {Object} dialogId
     * @returns {string}
     */
    handleModal: function(openStamp, closeStamp, dialogId) {
      
      if (openStamp === undefined && closeStamp === undefined) {
        return window.dash_clientside.no_update
      }

      if (closeStamp === undefined) {
        openDialog(dialogId)
      } else if (openStamp === undefined) {
        closeDialog(dialogId)
      } else if (openStamp > closeStamp) {
        openDialog(dialogId)
      } else if (closeStamp > openStamp) {
        closeDialog(dialogId)
      }
      return window.dash_clientside.no_update
    },
    updateQuoteGraph: function(plotType, relayoutData, startDate, endDate, figure) {

      if (figure === undefined) { return window.dash_clientside.no_update }

      const triggered_id = dash_clientside.callback_context.triggered_id
      console.log(triggered_id)

      if (triggered_id.component === "QuoteGraphTypeAIO") {
        figure = updateQuoteGraphType(figure, plotType)
      }
      if ((triggered_id.component === "QuoteGraphAIO") && (relayoutData !== undefined)) {
        figure = updateGraphRelayout(relayoutData, figure)
      }
      else if (triggered_id.component === "QuoteDatePickerAIO") {
        figure = quoteGraphRange(figure, startDate, endDate)
      }

      return {data: figure.data, layout: figure.layout}
    },
    updateQuoteDatePicker: function(figure) {
      if (figure === undefined) { return window.dash_clientside.no_update }

      const minDates = []
      const maxDates = []
      for (const data of figure.data) {
        const dates = data.x.map(dateString => new Date(dateString).getTime())
        minDates.push(Math.min(...dates))
        maxDates.push(Math.max(...dates))
      }

      const minDate = new Date(Math.min(...minDates))
      const maxDate = new Date(Math.max(...maxDates))
      return [minDate.toISOString(), maxDate.toISOString()]
    }
  }
})



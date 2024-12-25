const dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.FinancialsTooltip = (props) => {
  style = "p-2 border border-secondary rounded shadow bg-text/50 backdrop-blur-sm"

  return React.createElement("div", {
    className: props.className || style
  },
    React.createElement("h4", {}, props.labels[props.rowIndex])
  )
}

dagcomponentfuncs.TrendLine = (props) => {
  //const {setData} = props
  //function setProps() {
  //  const graphProps = arguments[0]
  //  if (graphProps["clickData"]) {
  //    setData(graphProps)
  //  }
  //}
  return React.createElement(window.dash_core_components.Graph, {
    figure: {
      data: [{
        x: props.value.x,
        y: props.value.y,
        type: "bar",
        orientation: 'v',
        //mode: 'lines',
        marker: { color: "#1f77b4" }
      }],
      layout: {
        showlegend: false,
        xaxis: {
          visible: false,
          showticklabels: false
        },
        yaxis: {
          visible: false,
          showticklabels: false
        },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        template: "plotly_white",
        autosize: true
      }
    },
    style: { height: "100%" },
    config: { displayModeBar: false, staticPlot: true }
  })
}

dagcomponentfuncs.CompanyLink = (props) => {
  const [display, link] = props.value.split("|", 2)

  return React.createElement("a", {
    href: `/company/${link}/overview`
  }, display)
}

/**
 * @param {number} value
 * @param {number} left
 * @param {string} color
 * @param {string} height
 * @return {ReactElement}
 */
function BarUnderMarker(value, left, color, height = "0.5rem") {

  return React.createElement("div", {
    key: `bar.marker.under.${color}`,
    className: "absolute top-1/2 -translate-x-1/2 flex flex-col place-items-center",
    style: { left: `${left}%` }
  }, [
    React.createElement("div", {
      key: `div.bar.marker.under.dot.${color}`,
      className: "size-1 rounded-full bg-text"
    }),
    React.createElement("div", {
      key: `div.bar.marker.under.line.${color}`,
      className: "w-0.5 -translate-x-1/4 border-r border-text",
      style: { height: height }
    }),
    React.createElement("div", { key: `bar.marker.under.label.${color}`, className: "text-center" },
      React.createElement("span", { style: { color: color } }, value.toFixed(2))
    )
  ])
}

/**
 * @param {number} value
 * @param {number} left
 * @param {string} color
 * @param {string} height
 * @return {ReactElement}
 */
function BarOverMarker(value, left, color, height = "0.5rem") {
  return React.createElement("div", {
    key: `bar.marker.over.${color}`,
    className: "absolute bottom-1/2 left-1/3 -translate-x-1/2 translate-y-[0.125rem] flex flex-col place-items-center",
    style: { left: `${left}%` }
  }, [
    React.createElement("div", { key: `bar.marker.over.label.${color}`, className: "text-center" },
      React.createElement("span", { style: { color: color } }, value.toFixed(2))
    ),
    React.createElement("div", {
      key: `div.bar.marker.over.line.${color}`,
      className: "w-0.5 -translate-x-1/4 border-r border-text",
      style: { height: height }
    }),
    React.createElement("div", {
      key: `div.bar.marker.over.dot.${color}`,
      className: "size-1 rounded-full bg-text"
    })
  ])
}

/**
 * @param {object} items
 * @return {ReactElement}
 */
function BarLegend(items) {
  const li = []
  for (const [label, color] of Object.entries(items)) {
    li.push(
      React.createElement("li", {
        key: `li.bar.${label}`,
        className: "flex items-center before:content-['•'] before:text-text",
        style: { color: color }
      }, label)
    )
  }
  return React.createElement("ul", {
    key: "ul.bar.legend",
    className: "flex gap-2 justify-between list-none"
  }, li)
}

dagcomponentfuncs.ScreenerTooltip = (props) => {

  if (props.location === "header") {
    return React.createElement("div", {
      className: "p-1 bg-primary rounded border border-secondary text-text text-xs"
    }, props.value)
  }

  const regex = /(?<=\()(.+)(?=\))/
  const ticker = props.data.company.split("|")[0].match(regex)[0].split(",")[0]

  const exchangeRange = props.exchangeMax - props.exchangeMin
  const companyLeft = ((props.value - props.exchangeMin) / exchangeRange) * 100
  const exchangeLeft = ((props.exchangeMean - props.exchangeMin) / exchangeRange) * 100
  const sectorMean = props.sectorMean[props.data.sector]
  const sectorLeft = ((sectorMean - props.exchangeMin) / exchangeRange) * 100
  const sectorBarLeft = ((props.sectorMin[props.data.sector] - props.exchangeMin) / exchangeRange) * 100
  const sectorBarWidth = ((props.sectorMax[props.data.sector] - props.sectorMin[props.data.sector]) / exchangeRange) * 100

  legend = {
    [props.exchange]: "orange",
    [props.data.sector]: "green",
    [ticker]: "red"
  }

  return React.createElement("div", {
    className: "w-44 h-[6.5rem] px-2 grid grid-rows-[1fr_auto] bg-primary rounded border border-secondary text-text text-xs font-bold"
  }, [
    React.createElement("div", {
      key: "div.bar",
      className: "grid grid-cols-[auto_1fr_auto] gap-1 content-center"
    }, [
      React.createElement("span", {
        key: "span.bar.min",
      }, props.exchangeMin.toFixed(2)),
      React.createElement("div", {
        key: "div.bar.wrapper",
        className: "relative size-full"
      }, [
        React.createElement("div", {
          key: "div.bar.exchange",
          className: "absolute top-1/2 w-full h-0.5",
          style: { backgroundColor: legend[props.exchange] }
        }),
        React.createElement("div", {
          key: "div.bar.sector",
          className: "absolute top-1/2 h-0.5",
          style: {
            left: `${sectorBarLeft}%`,
            width: `${sectorBarWidth}%`,
            backgroundColor: legend[props.data.sector]
          }
        }),
        BarUnderMarker(props.value, companyLeft, legend[ticker]),
        BarOverMarker(props.exchangeMean, exchangeLeft, legend[props.exchange], "1.25rem"),
        BarOverMarker(sectorMean, sectorLeft, legend[props.data.sector]),
      ]),
      React.createElement("span", {
        key: "span.bar.high",
      }, props.exchangeMax.toFixed(2)),
    ]
    ),
    BarLegend(legend)
  ])
}

dagcomponentfuncs.TaxonomyTooltip = (props) => {
  style = "p-2 border border-secondary rounded shadow bg-text/50 backdrop-blur-sm"

  const listItems = props.value.map((item, index) =>
    React.createElement("li", { key: index }, item)
  );

  return React.createElement("div", { className: style },
    React.createElement("ul", null, ...listItems)
  )
}
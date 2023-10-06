var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.FinancialsTooltip = (props) => {
  style = 'p-2 border border-secondary rounded shadow bg-text/50 backdrop-blur-sm'

  return React.createElement('div', {
      className: props.className || style
    },
    React.createElement('h4', {}, props.labels[props.rowIndex])
  )
}

dagcomponentfuncs.TrendLine = (props) => {
  //const {setData} = props
  //function setProps() {
  //  const graphProps = arguments[0]
  //  if (graphProps['clickData']) {
  //    setData(graphProps)
  //  }
  //}
  return React.createElement(window.dash_core_components.Graph, {
    figure: props.value,
    style: {height: '100%'},
    config: {displayModeBar: false, staticPlot: true}
  })
}
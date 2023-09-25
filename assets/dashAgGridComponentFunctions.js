var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.FinancialsTooltip = function(props) {
  style = 'p-2 border border-secondary rounded shadow bg-text/50 backdrop-blur-sm'

  return React.createElement(
    'div',
    {
      className: props.className || style
    },
    React.createElement('h4', {}, props.labels[props.rowIndex])
  )
}
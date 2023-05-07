window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.props.hideout
            const value = feature.properties[colorProp]

            if (value === null) {
                style.fillColor = colorscale[0]
                return style
            }

            for (let i = 0; i < classes.length; i++) {
                if (value > classes[i]) {
                    style.fillColor = colorscale[i]
                }
            }
            return style
        }
    }
});
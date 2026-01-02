/**
 * SHAP explanation visualization component.
 * Shows top-K features that influenced the prediction.
 */
export default function ShapChart({ explanation }) {
    if (!explanation || explanation.length === 0) {
        return null
    }

    // Find max absolute value for scaling
    const maxAbsValue = Math.max(...explanation.map(e => Math.abs(e.shap_value)))

    return (
        <div className="glass-card p-6 animate-fade-in">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <span>🔍</span>
                Why This Prediction?
            </h3>

            <p className="text-sm text-slate-400 mb-4">
                Features that most influenced the prediction.
                <span className="text-emerald-400"> Green</span> reduces dropout risk,
                <span className="text-red-400"> Red</span> increases it.
            </p>

            <div className="space-y-3">
                {explanation.map((item, index) => {
                    const barWidth = (Math.abs(item.shap_value) / maxAbsValue) * 100
                    const isPositive = item.shap_value > 0

                    return (
                        <div key={item.feature} className="group">
                            <div className="flex items-center justify-between text-sm mb-1">
                                <span className="text-slate-300 font-medium">
                                    {formatFeatureName(item.feature)}
                                </span>
                                <span className="text-slate-400">
                                    Value: <span className="text-white">{item.value.toFixed(2)}</span>
                                </span>
                            </div>

                            {/* SHAP bar */}
                            <div className="flex items-center gap-2">
                                {/* Negative side */}
                                <div className="flex-1 flex justify-end">
                                    {!isPositive && (
                                        <div
                                            className="h-6 bg-gradient-to-l from-emerald-500 to-emerald-600 rounded-l transition-all duration-500"
                                            style={{ width: `${barWidth}%` }}
                                        />
                                    )}
                                </div>

                                {/* Center line */}
                                <div className="w-px h-8 bg-slate-600" />

                                {/* Positive side */}
                                <div className="flex-1">
                                    {isPositive && (
                                        <div
                                            className="h-6 bg-gradient-to-r from-red-500 to-red-600 rounded-r transition-all duration-500"
                                            style={{ width: `${barWidth}%` }}
                                        />
                                    )}
                                </div>
                            </div>

                            {/* SHAP value label */}
                            <div className="text-xs text-slate-500 text-center mt-1">
                                SHAP: {item.shap_value > 0 ? '+' : ''}{item.shap_value.toFixed(4)}
                            </div>
                        </div>
                    )
                })}
            </div>

            {/* Legend */}
            <div className="mt-6 pt-4 border-t border-slate-700">
                <div className="flex justify-center gap-6 text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-3 rounded bg-emerald-500" />
                        <span>Reduces Dropout Risk</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-3 rounded bg-red-500" />
                        <span>Increases Dropout Risk</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
}

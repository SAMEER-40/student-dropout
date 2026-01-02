/**
 * Prediction result display card.
 * Shows outcome with color coding and confidence meter.
 */
export default function ResultCard({ result }) {
    const { prediction_label, probabilities, confidence } = result

    const outcomeConfig = {
        'Dropout': {
            emoji: '🔴',
            color: 'red',
            badgeClass: 'badge-dropout',
            bgClass: 'from-red-500/20 to-red-600/10',
            message: 'High risk of dropping out'
        },
        'Enrolled': {
            emoji: '🟡',
            color: 'amber',
            badgeClass: 'badge-enrolled',
            bgClass: 'from-amber-500/20 to-amber-600/10',
            message: 'Likely to continue enrollment'
        },
        'Graduate': {
            emoji: '🟢',
            color: 'emerald',
            badgeClass: 'badge-graduate',
            bgClass: 'from-emerald-500/20 to-emerald-600/10',
            message: 'Likely to graduate successfully'
        }
    }

    const config = outcomeConfig[prediction_label] || outcomeConfig['Enrolled']

    return (
        <div className="glass-card overflow-hidden animate-fade-in">
            {/* Header with gradient */}
            <div className={`p-6 bg-gradient-to-r ${config.bgClass}`}>
                <div className="text-center">
                    <div className="text-5xl mb-3 animate-glow inline-block rounded-full p-4">
                        {config.emoji}
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-1">
                        {prediction_label}
                    </h2>
                    <p className="text-slate-300 text-sm">
                        {config.message}
                    </p>
                </div>
            </div>

            {/* Confidence meter */}
            <div className="p-6">
                <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-400">Confidence</span>
                        <span className="text-white font-semibold">
                            {(confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="progress-bar">
                        <div
                            className={`progress-fill bg-gradient-to-r ${confidence > 0.7
                                    ? 'from-emerald-500 to-emerald-400'
                                    : confidence > 0.5
                                        ? 'from-amber-500 to-amber-400'
                                        : 'from-red-500 to-red-400'
                                }`}
                            style={{ width: `${confidence * 100}%` }}
                        />
                    </div>
                </div>

                {/* Probability breakdown */}
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    Probability Distribution
                </h3>
                <div className="space-y-3">
                    {Object.entries(probabilities).map(([label, prob]) => (
                        <div key={label} className="flex items-center gap-3">
                            <span className="text-sm w-20 text-slate-300">{label}</span>
                            <div className="flex-1 progress-bar">
                                <div
                                    className={`progress-fill ${label === 'Dropout' ? 'bg-red-500' :
                                            label === 'Enrolled' ? 'bg-amber-500' :
                                                'bg-emerald-500'
                                        }`}
                                    style={{ width: `${prob * 100}%` }}
                                />
                            </div>
                            <span className="text-sm text-white w-14 text-right">
                                {(prob * 100).toFixed(1)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

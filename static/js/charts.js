// Dashboard "scan confidence over time"
(function(){
    const el = document.getElementById("scanChart");
    if (!el || !window.DASHBOARD_CHART_DATA) return;

    const labels = window.DASHBOARD_CHART_DATA.map(p => p.t);
    const dataVals = window.DASHBOARD_CHART_DATA.map(p => Math.round(p.conf*100));

    new Chart(el, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Confidence %',
                data: dataVals,
                tension: 0.3,
                fill: true,
            }]
        },
        options: {
            responsive:true,
            plugins:{
                legend:{display:false},
                tooltip:{enabled:true}
            },
            scales:{
                y:{beginAtZero:true,max:100}
            }
        }
    });
})();

// Analysis timeline chart
(function(){
    const el = document.getElementById("timelineChart");
    if (!el || !window.ANALYSIS_TIMELINE) return;

    const labels = window.ANALYSIS_TIMELINE.map(p => p.t);
    const confVals = window.ANALYSIS_TIMELINE.map(p => Math.round(p.conf*100));

    new Chart(el, {
        type: 'bar',
        data:{
            labels,
            datasets:[{
                label:'Confidence %',
                data:confVals,
            }]
        },
        options:{
            responsive:true,
            plugins:{
                legend:{display:false},
                tooltip:{enabled:true}
            },
            scales:{
                y:{beginAtZero:true,max:100}
            }
        }
    });
})();

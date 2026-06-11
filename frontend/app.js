// SmartHouse AI Dashboard Application Logic
// Integrates with FastAPI endpoints and renders Plotly.js charts

document.addEventListener("DOMContentLoaded", () => {
    // Application State
    let activeTab = "overview";
    let filterCity = "All";
    let filterType = "All";
    let filterBeds = "All";
    
    let currentPredictionInput = null; // Store last predicted property specs for What-If simulator
    let cachedOverviewData = null;
    let cachedTrendsData = null;
    let cachedGeoData = null;

    // Initialize Lucide Icons
    lucide.createIcons();

    // DOM Elements
    const navButtons = document.querySelectorAll(".nav-item");
    const tabPanels = document.querySelectorAll(".tab-panel");
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");
    
    // Global Filters
    const citySelect = document.getElementById("filter-city");
    const typeSelect = document.getElementById("filter-type");
    const bedsSelect = document.getElementById("filter-beds");
    const btnExportCsv = document.getElementById("btn-export-csv");

    // Tab Navigation Routing
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabName = btn.getAttribute("data-tab");
            switchTab(tabName);
        });
    });

    function switchTab(tabName) {
        activeTab = tabName;
        
        // Update nav buttons active states
        navButtons.forEach(btn => {
            if (btn.getAttribute("data-tab") === tabName) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });

        // Update panels
        tabPanels.forEach(panel => {
            if (panel.id === `panel-${tabName}`) {
                panel.classList.add("active");
            } else {
                panel.classList.remove("active");
            }
        });

        // Update headers and trigger tab specific loads
        const titles = {
            overview: { title: "Executive Overview", subtitle: "Real estate analytics and key performance drivers" },
            predictor: { title: "Valuation Predictor", subtitle: "Smart valuation estimation and Explainable AI analysis" },
            trends: { title: "Market Trends & Forecasting", subtitle: "Historical transaction logs and 12-month appreciation projections" },
            geographic: { title: "Geographic Intelligence", subtitle: "Geospatial distribution and heatmaps of average listings" },
            investments: { title: "Investment Opportunities", subtitle: "Undervalued deals ranked by pricing discount ratio" },
            mlops: { title: "MLOps & Model Performance", subtitle: "Regression pipelines comparison and active drift monitoring" }
        };

        pageTitle.innerText = titles[tabName].title;
        pageSubtitle.innerText = titles[tabName].subtitle;

        // Fetch appropriate tab data
        loadTabContent(tabName);
    }

    function loadTabContent(tabName) {
        if (tabName === "overview") {
            loadOverviewData();
        } else if (tabName === "trends") {
            loadTrendsData();
        } else if (tabName === "geographic") {
            loadGeographicData();
        } else if (tabName === "investments") {
            loadInvestmentDeals();
        } else if (tabName === "mlops") {
            loadMLOpsMetrics();
        }
    }

    // Load Filter Dropdowns
    async function loadFilterMetadata() {
        try {
            const response = await fetch("/api/meta");
            const data = await response.json();
            
            // Populate City Select
            citySelect.innerHTML = data.cities.map(c => `<option value="${c}">${c === "All" ? "All Cities" : c + ", " + getCityState(c)}</option>`).join("");
            
            // Populate Type Select
            typeSelect.innerHTML = data.property_types.map(t => `<option value="${t}">${t === "All" ? "All Types" : t}</option>`).join("");
            
            // Populate Beds Select
            bedsSelect.innerHTML = data.bedrooms.map(b => `<option value="${b}">${b === "All" ? "All Bedrooms" : b + " Bedrooms"}</option>`).join("");
            
            // Wire filter triggers
            [citySelect, typeSelect, bedsSelect].forEach(select => {
                select.addEventListener("change", () => {
                    filterCity = citySelect.value;
                    filterType = typeSelect.value;
                    filterBeds = bedsSelect.value;
                    
                    // Refresh current active tab
                    loadTabContent(activeTab);
                });
            });
        } catch (error) {
            console.error("Error loading filter metadata:", error);
        }
    }

    function getCityState(city) {
        const states = { Seattle: "WA", Austin: "TX", Chicago: "IL", Orlando: "FL", Phoenix: "AZ" };
        return states[city] || "";
    }

    // ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
    async function loadOverviewData() {
        try {
            const response = await fetch(`/api/overview?city=${filterCity}&prop_type=${filterType}&bedrooms=${filterBeds}`);
            const data = await response.json();
            cachedOverviewData = data;

            // Update KPIs
            document.getElementById("kpi-listings").innerText = data.total_listings.toLocaleString();
            document.getElementById("kpi-avg-price").innerText = `$${Math.round(data.avg_price).toLocaleString()}`;
            document.getElementById("kpi-med-price").innerText = `$${Math.round(data.median_price).toLocaleString()}`;
            document.getElementById("kpi-sqft-price").innerText = `$${Math.round(data.avg_price_sqft).toLocaleString()}`;
            
            document.getElementById("kpi-market-growth").innerText = `${data.market_growth_pct}%`;
            document.getElementById("kpi-school").innerHTML = `<i data-lucide="graduation-cap"></i> School Rating Avg: <strong>${data.avg_school_rating}</strong>`;
            lucide.createIcons();

            // Populate Insights
            const insightsContainer = document.getElementById("insights-container");
            insightsContainer.innerHTML = data.insights.map(ins => `<li>${ins}</li>`).join("");

            // Render Distribution Plot
            loadOverviewDistributionPlot();
        } catch (error) {
            console.error("Error loading overview metrics:", error);
        }
    }

    async function loadOverviewDistributionPlot() {
        try {
            const response = await fetch(`/api/analytics?city=${filterCity}&prop_type=${filterType}&bedrooms=${filterBeds}`);
            const data = await response.json();

            const trace = {
                x: data.price_dist.bin_edges,
                y: data.price_dist.counts,
                type: 'bar',
                marker: {
                    color: '#10b981',
                    opacity: 0.8
                },
                name: 'Price'
            };

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#9ca3af', family: 'Inter, sans-serif' },
                xaxis: { title: 'Valuation ($)', gridcolor: '#243042', tickformat: '$,.0f' },
                yaxis: { title: 'Property Count', gridcolor: '#243042' },
                margin: { l: 50, r: 20, t: 30, b: 50 },
                bargap: 0.05
            };

            document.getElementById('overview-dist-chart').innerHTML = '';
            Plotly.newPlot('overview-dist-chart', [trace], layout, {responsive: true, displayModeBar: false});
        } catch (err) {
            console.error("Error rendering distribution plot:", err);
        }
    }

    // ----------------- SQL CONSOLE WORKSPACE -----------------
    const btnRunSql = document.getElementById("btn-run-sql");
    const sqlQuerySelect = document.getElementById("sql-query-select");
    const sqlCodeDisplay = document.getElementById("sql-code-display");
    const sqlTableContainer = document.getElementById("sql-table-container");

    sqlQuerySelect.addEventListener("change", fetchSelectedQueryCode);
    btnRunSql.addEventListener("click", runSelectedSqlQuery);

    async function fetchSelectedQueryCode() {
        const index = sqlQuerySelect.value;
        try {
            const response = await fetch(`/api/sql-analytics?query_index=${index}`);
            const data = await response.json();
            sqlCodeDisplay.innerText = data.raw_sql;
        } catch (err) {
            console.error("Error loading SQL code preview:", err);
        }
    }

    async function runSelectedSqlQuery() {
        const index = sqlQuerySelect.value;
        sqlTableContainer.innerHTML = `<div class="chart-loading">Running CTE window aggregations...</div>`;
        try {
            const response = await fetch(`/api/sql-analytics?query_index=${index}`);
            const data = await response.json();
            
            if (!data.results || data.results.length === 0) {
                sqlTableContainer.innerHTML = `<div class="empty-state">No results returned</div>`;
                return;
            }

            // Build table
            const headers = Object.keys(data.results[0]);
            let tableHtml = `<table class="custom-table"><thead><tr>`;
            headers.forEach(h => {
                // Human read headers format
                const cleanHeader = h.replace(/_/g, " ").toUpperCase();
                tableHtml += `<th>${cleanHeader}</th>`;
            });
            tableHtml += `</tr></thead><tbody>`;
            
            data.results.forEach(row => {
                tableHtml += `<tr>`;
                headers.forEach(h => {
                    let val = row[h];
                    if (typeof val === 'number') {
                        if (h.includes("price") || h.includes("value") || h.includes("discount")) {
                            val = `$${Math.round(val).toLocaleString()}`;
                        } else if (h.includes("pct") || h.includes("percentage")) {
                            val = `${val.toFixed(1)}%`;
                        } else if (val % 1 !== 0) {
                            val = val.toFixed(2);
                        }
                    }
                    tableHtml += `<td>${val}</td>`;
                });
                tableHtml += `</tr>`;
            });
            tableHtml += `</tbody></table>`;
            
            sqlTableContainer.innerHTML = tableHtml;
        } catch (err) {
            sqlTableContainer.innerHTML = `<div class="empty-state text-danger">Query Error: ${err.message}</div>`;
        }
    }

    // ----------------- TAB 2: VALUATION PREDICTOR -----------------
    const predictorForm = document.getElementById("predictor-form");
    const predResultsPanel = document.getElementById("prediction-results-panel");
    const valEstPrice = document.getElementById("val-est-price");
    const valPriceRange = document.getElementById("val-price-range");
    const valConfidence = document.getElementById("val-confidence");
    const valGrade = document.getElementById("val-grade");
    const valDeal = document.getElementById("val-deal");
    const positiveShapList = document.getElementById("positive-shap-list");
    const negativeShapList = document.getElementById("negative-shap-list");

    // Slider simulation elements
    const sliderArea = document.getElementById("slider-area");
    const valSliderArea = document.getElementById("val-slider-area");
    const sliderSchool = document.getElementById("slider-school");
    const valSliderSchool = document.getElementById("val-slider-school");
    const simAddPool = document.getElementById("sim-add-pool");
    const simAddGym = document.getElementById("sim-add-gym");
    const simMetroNear = document.getElementById("sim-metro-near");
    const simPriceDelta = document.getElementById("sim-price-delta");
    const simNewPrice = document.getElementById("sim-new-price");

    predictorForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Assemble payload
        const payload = {
            property_type: document.getElementById("pred-type").value,
            area_sqft: parseFloat(document.getElementById("pred-area").value),
            bedrooms: parseInt(document.getElementById("pred-beds").value),
            bathrooms: parseFloat(document.getElementById("pred-baths").value),
            floors: parseInt(document.getElementById("pred-floors").value),
            garage_spaces: parseInt(document.getElementById("pred-garage").value),
            year_built: parseInt(document.getElementById("pred-year").value),
            city: document.getElementById("pred-city").value,
            zip_code: document.getElementById("pred-zip").value,
            has_pool: document.getElementById("pred-has-pool").checked,
            has_garden: document.getElementById("pred-has-garden").checked,
            has_balcony: document.getElementById("pred-has-balcony").checked,
            has_parking: document.getElementById("pred-has-parking").checked,
            has_security: document.getElementById("pred-has-security").checked,
            has_gym: document.getElementById("pred-has-gym").checked,
            school_rating: parseFloat(document.getElementById("pred-school").value),
            crime_rate: parseFloat(document.getElementById("pred-crime").value),
            distance_to_metro: parseFloat(document.getElementById("pred-metro").value),
            distance_to_center: parseFloat(document.getElementById("pred-center").value),
            distance_to_airport: parseFloat(document.getElementById("pred-airport").value),
            nearby_hospitals: parseInt(document.getElementById("pred-hospitals").value),
            nearby_shopping: parseInt(document.getElementById("pred-shopping").value),
            list_price: document.getElementById("pred-list-price").value ? parseFloat(document.getElementById("pred-list-price").value) : null
        };

        currentPredictionInput = payload; // Store for What-If
        
        // Reset sliders
        sliderArea.value = 0;
        valSliderArea.innerText = "+0 sq ft";
        sliderSchool.value = 0;
        valSliderSchool.innerText = "+0.0 rating points";
        simAddPool.checked = false;
        simAddGym.checked = false;
        simMetroNear.checked = false;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            // Show Results
            valEstPrice.innerText = `$${Math.round(data.estimated_price).toLocaleString()}`;
            valPriceRange.innerText = `$${Math.round(data.price_range.lower).toLocaleString()} to $${Math.round(data.price_range.upper).toLocaleString()}`;
            valConfidence.innerText = `${data.confidence_score}%`;
            valGrade.innerText = `${data.property_rating} / 5`;
            valDeal.innerText = data.investment_rating;
            
            // Deal rating class colors
            if (data.investment_rating.includes("Buy")) {
                valDeal.className = "text-success";
            } else if (data.investment_rating.includes("Overvalued")) {
                valDeal.className = "text-danger";
            } else {
                valDeal.className = "text-amber";
            }

            // Populate Explainable AI Contributions
            positiveShapList.innerHTML = data.explainable_ai.positive_factors.map(f => `
                <li class="driver-item">
                    <span class="driver-label">${f.factor}</span>
                    <span class="driver-val text-success">+$${Math.round(f.impact).toLocaleString()}</span>
                </li>
            `).join("");

            negativeShapList.innerHTML = data.explainable_ai.negative_factors.map(f => `
                <li class="driver-item">
                    <span class="driver-label">${f.factor}</span>
                    <span class="driver-val text-danger">-$${Math.round(Math.abs(f.impact)).toLocaleString()}</span>
                </li>
            `).join("");

            // Setup Simulation output defaults
            simPriceDelta.innerText = "+$0";
            simPriceDelta.className = "text-success text-large";
            simNewPrice.innerText = `$${Math.round(data.estimated_price).toLocaleString()}`;

        } catch (err) {
            console.error("Prediction failed:", err);
        }
    });

    // Wire Scenario Sliders
    [sliderArea, sliderSchool, simAddPool, simAddGym, simMetroNear].forEach(input => {
        input.addEventListener("input", triggerWhatIfSimulation);
        input.addEventListener("change", triggerWhatIfSimulation);
    });

    async function triggerWhatIfSimulation() {
        if (!currentPredictionInput) return;

        const areaVal = parseInt(sliderArea.value);
        valSliderArea.innerText = `+${areaVal} sq ft`;

        const schoolVal = parseFloat(sliderSchool.value);
        valSliderSchool.innerText = `+${schoolVal.toFixed(1)} rating points`;

        const simPayload = {
            base_property: currentPredictionInput,
            area_change: areaVal,
            school_change: schoolVal,
            metro_station_added: simMetroNear.checked,
            add_pool: simAddPool.checked,
            add_gym: simAddGym.checked
        };

        try {
            const response = await fetch("/api/whatif", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(simPayload)
            });
            const data = await response.json();

            // Display sim deltas
            const delta = data.price_delta;
            if (delta >= 0) {
                simPriceDelta.innerText = `+$${Math.round(delta).toLocaleString()} (+${data.percentage_change}%)`;
                simPriceDelta.className = "text-success text-large";
            } else {
                simPriceDelta.innerText = `-$${Math.round(Math.abs(delta)).toLocaleString()} (${data.percentage_change}%)`;
                simPriceDelta.className = "text-danger text-large";
            }

            simNewPrice.innerText = `$${Math.round(data.simulated_price).toLocaleString()}`;
        } catch (err) {
            console.error("Scenario simulation failed:", err);
        }
    }

    // ----------------- TAB 3: MARKET TRENDS & FORECASTS -----------------
    async function loadTrendsData() {
        try {
            const response = await fetch("/api/trends");
            const data = await response.json();
            cachedTrendsData = data;

            renderHistoricalTrendsPlot(data);
            renderForecastTrendsPlot(data);
            renderAppreciationBarPlot(data);
        } catch (err) {
            console.error("Error loading trends charts:", err);
        }
    }

    function renderHistoricalTrendsPlot(data) {
        const hist = data.historical;
        const keys = Object.keys(hist).filter(k => k !== 'months');
        const colors = { global_avg: '#10b981', Seattle: '#3b82f6', Austin: '#f59e0b', Chicago: '#ef4444', Orlando: '#8b5cf6', Phoenix: '#ec4899' };

        const traces = keys.map(city => {
            return {
                x: hist.months,
                y: hist[city],
                type: 'scatter',
                mode: 'lines',
                name: city === 'global_avg' ? 'Global Average' : city,
                line: {
                    color: colors[city] || '#9ca3af',
                    width: city === 'global_avg' ? 3 : 1.5,
                    dash: city === 'global_avg' ? 'solid' : 'dot'
                }
            };
        });

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', family: 'Inter, sans-serif' },
            xaxis: { gridcolor: '#243042' },
            yaxis: { gridcolor: '#243042', tickformat: '$,.0f' },
            margin: { l: 65, r: 20, t: 20, b: 95 },
            legend: { orientation: 'h', y: -0.2, yanchor: 'bottom', x: 0.5, xanchor: 'center' }
        };

        document.getElementById('trends-historical-plot').innerHTML = '';
        Plotly.newPlot('trends-historical-plot', traces, layout, {responsive: true, displayModeBar: false});
    }

    function renderForecastTrendsPlot(data) {
        const fore = data.forecast;
        const keys = Object.keys(fore).filter(k => k !== 'months');
        const colors = { global_avg: '#10b981', Seattle: '#3b82f6', Austin: '#f59e0b', Chicago: '#ef4444', Orlando: '#8b5cf6', Phoenix: '#ec4899' };

        // Prepend last historical data point for visual continuity
        const lastHistIdx = data.historical.months.length - 1;
        const lastMonth = data.historical.months[lastHistIdx];
        
        const xaxis = [lastMonth].concat(fore.months);

        const traces = keys.map(city => {
            const firstPoint = data.historical[city][lastHistIdx];
            const yvals = [firstPoint].concat(fore[city]);
            
            return {
                x: xaxis,
                y: yvals,
                type: 'scatter',
                mode: 'lines+markers',
                name: city === 'global_avg' ? 'Global Projected' : city,
                line: {
                    color: colors[city] || '#9ca3af',
                    width: city === 'global_avg' ? 3 : 1.5
                }
            };
        });

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', family: 'Inter, sans-serif' },
            xaxis: { gridcolor: '#243042' },
            yaxis: { gridcolor: '#243042', tickformat: '$,.0f' },
            margin: { l: 65, r: 20, t: 20, b: 95 },
            legend: { orientation: 'h', y: -0.2, yanchor: 'bottom', x: 0.5, xanchor: 'center' }
        };

        document.getElementById('trends-forecast-plot').innerHTML = '';
        Plotly.newPlot('trends-forecast-plot', traces, layout, {responsive: true, displayModeBar: false});
    }

    function renderAppreciationBarPlot(data) {
        const cities = Object.keys(data.forecast).filter(k => k !== 'months' && k !== 'global_avg');
        
        const growthRates = cities.map(city => {
            const histValues = data.historical[city];
            const startPrice = histValues[histValues.length - 1]; // Current price
            const endPrice = data.forecast[city][11]; // Price in 12 months
            const rate = ((endPrice - startPrice) / startPrice) * 100;
            return { city, rate };
        });

        // Sort by growth rate
        growthRates.sort((a, b) => b.rate - a.rate);

        const trace = {
            x: growthRates.map(x => x.city),
            y: growthRates.map(x => x.rate),
            type: 'bar',
            marker: {
                color: growthRates.map((x, i) => i === 0 ? '#10b981' : '#3b82f6'),
                opacity: 0.8
            }
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', family: 'Inter, sans-serif' },
            xaxis: { gridcolor: '#243042' },
            yaxis: { title: 'Projected 12M Growth Rate (%)', gridcolor: '#243042', ticksuffix: '%' },
            margin: { l: 50, r: 20, t: 20, b: 40 }
        };

        document.getElementById('trends-growth-bar').innerHTML = '';
        Plotly.newPlot('trends-growth-bar', [trace], layout, {responsive: true, displayModeBar: false});
    }

    // ----------------- TAB 4: GEOGRAPHIC INTELLIGENCE -----------------
    async function loadGeographicData() {
        try {
            const response = await fetch("/api/geographic");
            const data = await response.json();
            cachedGeoData = data;

            renderGeospatialScatterPlot(data);
        } catch (err) {
            console.error("Error loading geographic metrics:", err);
        }
    }

    function renderGeospatialScatterPlot(data) {
        // Plotly scattermapbox representing geographical locations colored by price
        const points = data.map_points;

        const trace = {
            lon: points.map(p => p.longitude),
            lat: points.map(p => p.latitude),
            text: points.map(p => `${p.property_id} (${p.property_type})<br>Price: $${Math.round(p.sale_price).toLocaleString()}<br>Zip: ${p.zip_code}`),
            mode: 'markers',
            marker: {
                size: 6,
                color: points.map(p => p.sale_price),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Valuation ($)',
                    titleside: 'top',
                    tickformat: '$,.0f',
                    x: 0.95,
                    y: 0.5,
                    thickness: 15,
                    len: 0.8
                }
            },
            type: 'scattermapbox'
        };

        const layout = {
            mapbox: {
                style: 'carto-darkmatter',
                center: { lat: 38.5, lon: -98.0 },
                zoom: 3.5
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', family: 'Inter, sans-serif' },
            margin: { l: 0, r: 0, t: 0, b: 0 }
        };

        document.getElementById('geo-map-plot').innerHTML = '';
        Plotly.newPlot('geo-map-plot', [trace], layout, {responsive: true});
    }

    // ----------------- TAB 5: INVESTMENT DEALS ANALYZER -----------------
    async function loadInvestmentDeals() {
        const body = document.getElementById("deals-table-body");
        body.innerHTML = `<tr><td colspan="10" class="loading-td">Analyzing transaction undervalued differentials...</td></tr>`;
        
        try {
            const response = await fetch(`/api/investments?city=${filterCity}&prop_type=${filterType}`);
            const data = await response.json();
            
            if (!response.ok) {
                const errMsg = data.detail || "Server error";
                body.innerHTML = `<tr><td colspan="10" class="loading-td text-danger">Error retrieving deals: ${errMsg}</td></tr>`;
                return;
            }
            
            if (!Array.isArray(data)) {
                body.innerHTML = `<tr><td colspan="10" class="loading-td text-danger">Error retrieving deals: Invalid response format</td></tr>`;
                return;
            }
            
            if (data.length === 0) {
                body.innerHTML = `<tr><td colspan="10" class="loading-td">No undervalued opportunities detected. Check ML calibration.</td></tr>`;
                return;
            }

            body.innerHTML = data.map(deal => `
                <tr>
                    <td><strong>${deal.property_id}</strong></td>
                    <td>${deal.city} (${deal.zip_code})</td>
                    <td>${deal.property_type}</td>
                    <td>${deal.area_sqft.toLocaleString()} sq ft</td>
                    <td>${deal.bedrooms}b / ${deal.bathrooms}ba</td>
                    <td>$${Math.round(deal.market_price).toLocaleString()}</td>
                    <td>$${Math.round(deal.predicted_price).toLocaleString()}</td>
                    <td class="text-success">+$${Math.round(deal.discount_value).toLocaleString()}</td>
                    <td class="text-success"><strong>${deal.discount_percentage.toFixed(1)}%</strong></td>
                    <td>
                        <button class="btn btn-secondary btn-run-deal" data-json='${JSON.stringify(deal)}'>
                            Analyze Deal
                        </button>
                    </td>
                </tr>
            `).join("");

            // Setup deal analyzer click triggers
            document.querySelectorAll(".btn-run-deal").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    const jsonStr = btn.getAttribute("data-json");
                    const deal = JSON.parse(jsonStr);
                    loadDealInPredictor(deal);
                });
            });

        } catch (err) {
            body.innerHTML = `<tr><td colspan="10" class="loading-td text-danger">Error retrieving deals: ${err.message}</td></tr>`;
        }
    }

    // Prefill form from deal details
    function loadDealInPredictor(deal) {
        switchTab("predictor");
        // Prefill city
        document.getElementById("pred-city").value = deal.city;
        document.getElementById("pred-zip").value = deal.zip_code;
        document.getElementById("pred-type").value = deal.property_type;
        document.getElementById("pred-area").value = deal.area_sqft;
        document.getElementById("pred-beds").value = deal.bedrooms;
        document.getElementById("pred-baths").value = deal.bathrooms;
        document.getElementById("pred-list-price").value = deal.market_price;
        
        // Auto trigger submit
        document.getElementById("predictor-form").dispatchEvent(new Event("submit"));
    }

    // ----------------- TAB 6: MODEL PERFORMANCE & MLOPS -----------------
    async function loadMLOpsMetrics() {
        loadModelDriftTracker();
        loadMLModelComparisonTable();
    }

    async function loadMLModelComparisonTable() {
        const body = document.getElementById("ml-models-body");
        body.innerHTML = `<tr><td colspan="6" class="loading-td">Retrieving MLflow models logs...</td></tr>`;
        
        try {
            const response = await fetch("/api/models");
            const data = await response.json();
            
            // Populate table
            body.innerHTML = data.comparison.map((m, idx) => `
                <tr class="${idx === 6 ? 'card-highlight' : ''}">
                    <td><strong>${m.Model}</strong></td>
                    <td>$${Math.round(m.MAE).toLocaleString()}</td>
                    <td>$${Math.round(m.RMSE).toLocaleString()}</td>
                    <td><strong>${m.R2.toFixed(4)}</strong></td>
                    <td>${m.MAPE.toFixed(2)}%</td>
                    <td>
                        <span class="badge ${idx === 6 ? 'badge-emerald' : 'text-muted'}">
                            ${idx === 6 ? 'Active / Registered' : 'Shadow'}
                        </span>
                    </td>
                </tr>
            `).join("");

            // Render Global SHAP Chart
            renderShapImportancePlot(data.feature_importances);

        } catch (err) {
            body.innerHTML = `<tr><td colspan="6" class="loading-td text-danger">Error loading ML metrics: ${err.message}</td></tr>`;
        }
    }

    function renderShapImportancePlot(importances) {
        // Horizontal bar plot
        const trace = {
            y: importances.map(x => x.feature).reverse(),
            x: importances.map(x => x.importance).reverse(),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: '#10b981',
                opacity: 0.8
            }
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', family: 'Inter, sans-serif' },
            xaxis: { title: 'Global Mean (|SHAP|) Relative Importance (%)', gridcolor: '#243042', ticksuffix: '%' },
            yaxis: { gridcolor: '#243042' },
            margin: { l: 150, r: 20, t: 10, b: 40 }
        };

        document.getElementById('ml-shap-plot').innerHTML = '';
        Plotly.newPlot('ml-shap-plot', [trace], layout, {responsive: true, displayModeBar: false});
    }

    async function loadModelDriftTracker() {
        const statusVal = document.getElementById("drift-overall-status");
        const countVal = document.getElementById("drift-logged-count");
        const container = document.getElementById("drift-features-container");
        
        container.innerHTML = `<div class="chart-loading">Analyzing drifts...</div>`;

        try {
            const response = await fetch("/api/drift");
            const data = await response.json();
            
            countVal.innerText = data.logged_count;
            statusVal.innerText = data.status;
            
            if (data.status === "Healthy") {
                statusVal.className = "drift-status-val text-success";
            } else {
                statusVal.className = "drift-status-val text-danger";
            }

            if (data.logged_count < 10) {
                container.innerHTML = `<div class="empty-state">Logged predictions profile list is building (minimum 10 predictions).</div>`;
                return;
            }

            container.innerHTML = data.features_drift.map(f => `
                <li class="drift-feature-item">
                    <div>
                        <div class="drift-name">${f.feature.replace(/_/g, " ").toUpperCase()}</div>
                        <div class="drift-vals">Baseline Avg: ${f.training_average} | Predict Avg: ${f.prediction_average}</div>
                    </div>
                    <div class="flex-row" style="align-items: center; gap: 12px;">
                        <span class="drift-vals">Dev: ${f.deviation_pct}%</span>
                        <span class="drift-badge-status ${f.status.toLowerCase().includes("warning") ? 'warning' : 'stable'}">
                            ${f.status}
                        </span>
                    </div>
                </li>
            `).join("");

        } catch (err) {
            container.innerHTML = `<div class="empty-state text-danger">Error tracking drift: ${err.message}</div>`;
        }
    }

    // Wire Retraining Pipeline Button
    const btnRetrain = document.getElementById("btn-trigger-retrain");
    const retrainStatus = document.getElementById("retrain-status");

    btnRetrain.addEventListener("click", async () => {
        btnRetrain.disabled = true;
        retrainStatus.innerText = "Queued background retraining process...";
        
        try {
            const response = await fetch("/api/retrain", { method: "POST" });
            const data = await response.json();
            retrainStatus.innerText = "ML retraining pipeline executing in background. Reloading metrics in 15 seconds...";
            
            // Fetch updates after 15 seconds
            setTimeout(() => {
                loadMLOpsMetrics();
                btnRetrain.disabled = false;
                retrainStatus.innerText = "Models updated. Ready to trigger next run.";
            }, 15000);

        } catch (err) {
            retrainStatus.innerText = `Retraining trigger error: ${err.message}`;
            btnRetrain.disabled = false;
        }
    });

    // ----------------- GLOBAL EXPORT & REPORTS -----------------
    btnExportCsv.addEventListener("click", () => {
        window.location.href = `/api/export?city=${filterCity}&prop_type=${filterType}&bedrooms=${filterBeds}&format=csv`;
    });

    // printable PDF Modal logic
    const modal = document.getElementById("report-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnCancelReport = document.getElementById("btn-cancel-report");
    const btnPrintReport = document.getElementById("btn-print-report");
    const modalContent = document.getElementById("modal-report-content");

    // Add print action button in predicted panel dynamically
    const resultsPanel = document.getElementById("prediction-results-panel");
    const exportDiv = document.createElement("div");
    exportDiv.className = "margin-top";
    exportDiv.innerHTML = `
        <button class="btn btn-secondary btn-full" id="btn-open-pdf-report">
            <i data-lucide="file-text"></i>
            Export PDF Valuation Report
        </button>
    `;
    resultsPanel.appendChild(exportDiv);
    lucide.createIcons();

    document.getElementById("btn-open-pdf-report").addEventListener("click", openPrintableReportModal);
    btnCloseModal.addEventListener("click", () => modal.style.display = "none");
    btnCancelReport.addEventListener("click", () => modal.style.display = "none");
    btnPrintReport.addEventListener("click", () => {
        window.print();
    });

    function openPrintableReportModal() {
        if (!currentPredictionInput) return;

        const estVal = document.getElementById("val-est-price").innerText;
        const estRange = document.getElementById("val-price-range").innerText;
        const estConfidence = document.getElementById("val-confidence").innerText;
        const estGrade = document.getElementById("val-grade").innerText;
        const estDeal = document.getElementById("val-deal").innerText;

        modalContent.innerHTML = `
            <div class="report-header-print">
                <div class="report-title-print">SmartHouse AI Valuation Report</div>
                <div class="report-date-print">Generated Date: ${new Date().toLocaleDateString()} | System Baseline: June 2026</div>
            </div>
            
            <div class="report-valuation-block">
                <div class="report-valuation-title">AI-Estimated Market Value</div>
                <div class="report-valuation-price">${estVal}</div>
                <div class="report-valuation-range">Valuation Range: ${estRange}</div>
            </div>

            <div class="report-section">
                <div class="report-section-title">Property Details</div>
                <div class="report-row-print"><span class="report-label-print">Location:</span><span class="report-val-print">${currentPredictionInput.city}, ${getCityState(currentPredictionInput.city)} (${currentPredictionInput.zip_code})</span></div>
                <div class="report-row-print"><span class="report-label-print">Property Type:</span><span class="report-val-print">${currentPredictionInput.property_type}</span></div>
                <div class="report-row-print"><span class="report-label-print">Size (Square Feet):</span><span class="report-val-print">${currentPredictionInput.area_sqft.toLocaleString()} sq ft</span></div>
                <div class="report-row-print"><span class="report-label-print">Beds / Baths / Garage:</span><span class="report-val-print">${currentPredictionInput.bedrooms} Bed / ${currentPredictionInput.bathrooms} Bath / ${currentPredictionInput.garage_spaces} Garage</span></div>
                <div class="report-row-print"><span class="report-label-print">Year Built:</span><span class="report-val-print">${currentPredictionInput.year_built}</span></div>
            </div>

            <div class="report-section">
                <div class="report-section-title">Neighborhood Quality Metrics</div>
                <div class="report-row-print"><span class="report-label-print">School Rating Score (1-10):</span><span class="report-val-print">${currentPredictionInput.school_rating} / 10.0</span></div>
                <div class="report-row-print"><span class="report-label-print">Crime Rating Score (1-10):</span><span class="report-val-print">${currentPredictionInput.crime_rate} / 10.0</span></div>
                <div class="report-row-print"><span class="report-label-print">Metro Station Distance:</span><span class="report-val-print">${currentPredictionInput.distance_to_metro} km</span></div>
                <div class="report-row-print"><span class="report-label-print">City Center Distance:</span><span class="report-val-print">${currentPredictionInput.distance_to_center} km</span></div>
            </div>

            <div class="report-section">
                <div class="report-section-title">Model Valuation Meta</div>
                <div class="report-row-print"><span class="report-label-print">Calibration Confidence Score:</span><span class="report-val-print">${estConfidence}</span></div>
                <div class="report-row-print"><span class="report-label-print">Property Location Grade:</span><span class="report-val-print">${estGrade}</span></div>
                <div class="report-row-print"><span class="report-label-print">Market Opportunity Deal Rating:</span><span class="report-val-print">${estDeal}</span></div>
            </div>

            <div style="font-size: 10px; color: #9ca3af; text-align: center; margin-top: 30px; border-top: 1px dashed #e5e7eb; padding-top: 10px;">
                CONFIDENTIAL: Real estate valuations generated by SmartHouse AI ensemble models are mathematical estimates. Verify structural details independently.
            </div>
        `;

        modal.style.display = "flex";
    }

    // --- On-load bootstrapping ---
    loadFilterMetadata().then(() => {
        loadTabContent("overview");
        fetchSelectedQueryCode();
    });
});

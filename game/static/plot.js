function makePlots(table_name, xLabel, maxUpper, speechID) {

  var margin = {top: 30, right: 50, bottom: 30, left: 75},
      width = 800 - margin.left - margin.right,
      height = 500 - margin.top - margin.bottom;

  var x = d3.scale.ordinal().rangeBands([0, width], .2),
      y = d3.scale.linear().range([height, 0]);

  var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([]);

  var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .ticks(10);

  var svg = d3.select(".graph")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg.append("text")
    .attr("class", "text-sm")
    .attr("x", 0)
    .attr("y", -25)
    .attr("dy", ".71em")
    .style("text-anchor", "middle")
    .text("Obama");

  svg.append("text")
    .attr("class", "text-sm")
    .attr("x", 0)
    .attr("y", height+10)
    .attr("dy", ".71em")
    .style("text-anchor", "middle")
    .text("Bush");

  svg.append("text")
    .attr("class", "text-sm")
    .attr("transform", "rotate(-90)")
    .attr("y", -margin.left)
    .attr("x", -1*height/2)
    .attr("dy", ".71em")
    .style("text-anchor", "middle")
    .text(xLabel);

  var order = $('order'),
      type = $('type'), 
      text = $('text'),
      lower = $("lower"),
      upper = $('upper');

  order.addEventListener("change", inputHandler);
  type.addEventListener("change", inputHandler);
  text.addEventListener("keyup", inputHandler);
  lower.addEventListener("change", inputHandler);
  upper.addEventListener("change", inputHandler);

  function inputHandler() {
    var upperVal = Math.min(Math.max(lower.value, upper.value, 1), maxUpper),
        lowerVal = Math.max(Math.min(lower.value, upper.value, maxUpper), 1);

    lower.value = lowerVal;
    upper.value = upperVal;

    getValues(lowerVal, upperVal, text.value, type.value, order.value);
  }

  function getValues(lowerVal, upperVal, textVal, typeVal, orderVal) {
    var http = new XMLHttpRequest(),
        url = "/values?" + [
          ["table", table_name].join("="),
          ["speech_id", (speechID || "")].join("="),
          ["type", typeVal].join("="),
          ["search", textVal].join("="),
          ["order", orderVal].join("="),
          ["from", lowerVal].join("="),
          ["to", upperVal].join("=")
        ].join("&");

    http.open("GET", url);
    http.addEventListener("load", updateGraph);
    http.send();
  }

  function updateGraph(res) {
    var data = JSON.parse(res.target.response),
        featureNames = data.feature_names,
        values = data.values,
        mag = data.mag;

    if (upper.value - lower.value >= values.length) {
      upper.classList.add("fadded");
    } else {
      upper.classList.remove("fadded");
    }

    drawUpdate(featureNames, values, mag);
  }

  function drawUpdate(featureNames, values, mag) {
    svg.select(".x.axis").remove(); 
    svg.select(".y.axis").remove(); 

    x.domain(featureNames);
    y.domain([-1*mag, mag]);

    var tip = d3.tip()
      .offset(function(d) {
        if (d >= 0) return [-10, 0];
        else return [y(d) - y(0) + margin.top + margin.bottom + 10, 0]; 
      })
      .html(function(d, i) { 
        return [
          "<div class='name'>", featureNames[i], "</div>",
          "<div class='value'>", d, "</div>"
        ].join(""); 
      });

    svg.call(tip);  

    var bars = svg.selectAll(".blue-bar, .red-bar")
      .data(values, function(d) { return d; });

    bars.enter().append("rect")
      .attr("class", function(d) { return (d >= 0 ? "blue-bar" : "red-bar"); })
      .on('mouseover', function(d, i) { 
          var tt;
          if (d >= 0) tt = "d3-tip-bot";
          else tt = "d3-tip-top"; 
          tip.attr("class", tt); 
          tip.show(d, i); 
        })
      .on('mouseout', tip.hide);    

    bars.exit().remove();

    bars
      .transition().duration(750) 
      .attr("x", function(d, i) { return x(featureNames[i]); })
      .attr("width", x.rangeBand())
      .attr("y", function(d) { return Math.min(y(0), y(d)); })
      .attr("height", function(d) { return Math.abs(y(0) - y(d)); }); 

    svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + y(0) + ")")
      .call(xAxis);

    svg.append("g")
      .attr("class", "y axis")
      .call(yAxis); 
  }

  inputHandler();

}

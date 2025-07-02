def get_html_repr(params):
    # Returns html representation of params

    def eq_to_html(eq):
        eq = re.sub(r"_(\d+)", r"<sub>\1</sub>", eq)
        eq = eq.replace("<=", "≤")
        eq = eq.replace(">=", "≥")
        eq = eq.replace("==", "=")
        for key, value in Greek.items():
            eq = eq.replace(key, value)
        return eq

    body = {
        "table1": {
            "RateParam": "",
            "TimeParam": "",
            "SizeParam": "",
            "ProportionParam": "",
        },
        "table2": "",
    }
    param_types = [
        "SizeParam",
        "RateParam",
        "ProportionParam",
        "TimeParam",
        "FixedParam",
    ]
    Greek = dict(zip(["eta", "rho", "pi", "tau"], ["&#951", "&#961", "&#960", "&#964"]))
    keys = list(params.keys())

    body = {"table1": {}}

    styles = {
        "table1": [
            "text-align:left; font-family:'Lucida Console'",
            "font-family:'Lucida Console'",
            "",
        ],
        "table2": [
            "text-align:left; font-family:'Lucida Console', monospace",
            "font-family:'Lucida Console'",
        ],
        "table_eq": ["text-align:left; font-family:'Lucida Console', monospace"],
    }

    table2_vals = []
    for param_type in param_types:
        table1_vals = []
        for key in keys:
            cur = params[key]
            if cur.__class__.__name__ == param_type:
                i = int(re.findall(r"\d+", key)[0])
                key_greek = re.findall(r"[a-z]+", key)[0]
                name = Greek[key_greek] + f"<sub>{i}</sub>"
                num = "{num:.3g}".format(num=cur.value)
                a = [name, num]
                if cur.__class__ is not FixedParam:
                    if params[key].train:
                        train = "&#9989"
                    else:
                        train = "&#10060"
                    a.append(train)
                table1_vals.append(a)
                for desc in cur.paths.values():
                    table2_vals.append([desc, name])
        table1_vals = sorted(table1_vals, key=lambda x: x[0])
        body["table1"][param_type] = get_body(table1_vals, styles=styles["table1"])

        table2_vals = sorted(table2_vals, key=lambda x: x[1])
    body["table2"] = get_body(table2_vals, styles=styles["table2"])

    eq_table = []
    # for eq in params.constraints.user_constraints.keys():
    #     eq_table.append([eq_to_html(eq)])
    eq_table = get_body(eq_table, styles=styles["table_eq"])

    # FIXME: Add the actual link to paper
    return f"""
<div style="display: inline-block; width: 30%;">
    <a href="https://github.com/jthlab/momi3" target="_blank">SOURCE CODE</a>
    <a href="https://www.biorxiv.org/content/10.1101/2024.03.26.586844v1" target="_blank">PAPER</a>
    <br>
    <img src="https://enesdilber.github.io/momilogo.png" style="width:75px;height:52px;">
    <table border="1" style="width: 100%;">
    <caption><h4>Size Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body["table1"]["SizeParam"]}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Rate Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body["table1"]["RateParam"]}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Proportion Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body["table1"]["ProportionParam"]}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Time Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
            <th >Infer</th>
        </tr>
    </thead>
    <tbody>
    {body["table1"]["TimeParam"]}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Fixed Parameters</h4></caption>
    <thead>
        <tr style="text-align: right;">
            <th >Parameter</th>
            <th >Value</th>
        </tr>
    </thead>
    <tbody>
    {body["table1"]["FixedParam"]}
    </tbody>
    </table>
    <table border="1" style="width: 100%;">
    <caption><h4>Constraints</h4></caption>
    <thead>
        <tr style="text-align: left;">
            <th style="text-align:left;">User Constraints</th>
        </tr>
    </thead>
    <tbody>
    {eq_table}
    </tbody>
    </table>
</div>
<div style="display: inline-block; width: 50%;">
<br>
    <table border="1" style="width: 100%;">
    <caption><h4>Parameter Locations</h4></caption>
    <thead>
        <tr>
            <th style="text-align:left; width:80%">Demes Parameter</th>
            <th >Parameter</th>
        </tr>
    </thead>
    <tbody>
    {body["table2"]}
    </tbody>
    </table>
</div>
"""

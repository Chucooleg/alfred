import argparse
import json
from pathlib import Path
from string import Template

from tqdm import tqdm

template = Template(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="description" content="$title">
    <meta name="author" content="Legg Yeung, Yonatan Bisk, Alex Polozov">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>$title</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" 
          integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" 
          integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
    <style>
        footer { font-size: 80%; padding: 1rem; margin-top: 1rem; background: #dedede; }
    </style>
</head>
<body>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" 
            integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" 
            integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" 
            integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script>
            
    <div class="container">
        <h1 class="display-4">Welcome</h1>
        <p class="lead"> Thank you for your help in this study. </p>
        <p class="lead"> Please watch the video below. It shows an agent performing a common household task in a 
        virtual room. After the video, you will see the Instructions that describe the agent's actions. Please 
        evaluate these Instructions by answering four questions at the bottom of this page, and click "Save the 
        results". Save the generated file and send it to the researcher who invited you to the study.</p>
        
        <h2>The agent's actions</h2>
        <video class="my-4" src="$video" width="512" height="512" controls></video>
        
        <h2>The agent's instructions</h2>
        <ol>$instructions</ol>
        
        <form id="likert">
            <div class="my-4 form-row">
                <h4>Q1: The instructions accurately describe the actions in the video.</h4>
                <div class="py-2" id="q1">
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q1Radio" value="-2" id="q1_opt-2">
                        <label for="q1_opt-2" class="form-check-label">Strongly disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q1Radio" value="-1" id="q1_opt-1">
                        <label for="q1_opt-1" class="form-check-label">Disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q1Radio" value="0" id="q1_opt0">
                        <label for="q1_opt0" class="form-check-label">Neither agree nor disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q1Radio" value="1" id="q1_opt1">
                        <label for="q1_opt1" class="form-check-label">Agree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q1Radio" value="2" id="q1_opt2">
                        <label for="q1_opt2" class="form-check-label">Strongly agree</label>
                    </div>
                </div>
            </div>
            
            <div class="my-4 form-row">
                <h4>Q2: The instructions are written in readable and understandable English.</h4>
                <div class="py-2" id="q2">
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q2Radio" value="-2" id="q2_opt-2">
                        <label for="q2_opt-2" class="form-check-label">Strongly disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q2Radio" value="-1" id="q2_opt-1">
                        <label for="q2_opt-1" class="form-check-label">Disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q2Radio" value="0" id="q2_opt0">
                        <label for="q2_opt0" class="form-check-label">Neither agree nor disagree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q2Radio" value="1" id="q2_opt1">
                        <label for="q2_opt1" class="form-check-label">Agree</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q2Radio" value="2" id="q2_opt2">
                        <label for="q2_opt2" class="form-check-label">Strongly agree</label>
                    </div>
                </div>
            </div>
            
            <div class="my-4 form-row">
                <h4>Q3: Please evaluate how natural the instructions are.</h4>
                <div class="py-2" id="q3">
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q3Radio" value="-2" id="q3_opt-2">
                        <label for="q3_opt-2" class="form-check-label">Robotic</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q3Radio" value="-1" id="q3_opt-1">
                        <label for="q3_opt-1" class="form-check-label">Somewhat robotic</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q3Radio" value="0" id="q3_opt0">
                        <label for="q3_opt0" class="form-check-label">Neutral</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q3Radio" value="1" id="q3_opt1">
                        <label for="q3_opt1" class="form-check-label">Somewhat natural</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q3Radio" value="2" id="q3_opt2">
                        <label for="q3_opt2" class="form-check-label">Natural</label>
                    </div>
                </div>
            </div>
            
            <div class="my-4 form-row">
                <h4>Q4: Please evaluate how concise the instructions are.</h4>
                <div class="py-2" id="q4">
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q4Radio" value="-2" id="q4_opt-2">
                        <label for="q4_opt-2" class="form-check-label">Concise</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q4Radio" value="-1" id="q4_opt-1">
                        <label for="q4_opt-1" class="form-check-label">Somewhat concise</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q4Radio" value="0" id="q4_opt0">
                        <label for="q4_opt0" class="form-check-label">Neutral</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q4Radio" value="1" id="q4_opt1">
                        <label for="q4_opt1" class="form-check-label">Somewhat verbose</label>
                    </div>
                    <div class="form-check form-check-inline">
                        <input required type="radio" class="form-check-input" name="q4Radio" value="2" id="q4_opt2">
                        <label for="q4_opt2" class="form-check-label">Verbose</label>
                    </div>
                </div>
            </div>
            
            <div class="my-4 form-row">
                <div class="text-center">
                    <button class="btn btn-outline-primary btn-lg" id="btnSend" type="submit">
                        <i class="fa fa-download" aria-hidden="true"></i> Save the results
                    </button>
                </div>
            </div>
        </form>
    </div>
    
    <div id="consentModal" class="modal fade" tabindex="-1" data-backdrop="static" data-keyboard="false">
        <div class="modal-dialog modal-xl modal-dialog-scrollable modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Microsoft Research Project Participation Consent Form</h5>
                </div>
                <div class="modal-body">
                    <h5>INTRODUCTION</h5>
                    <p>Thank you for taking the time to consider volunteering in a Microsoft Corporation research 
                    project. This form explains what would happen if you join this research project. Please read it 
                    carefully and take as much time as you need. Email the study team to ask about anything that is not clear. Participation in this study is voluntary and you may withdraw at any time.</p>
                    <h5>TITLE OF RESEARCH PROJECT</h5>
                    <p>Alfred Speaks: Instruction Generation for Embodied Agents</p>
                    <h5>PURPOSE</h5>
                    <p>The goal of this study is to evaluate the accuracy and fluency of auto-generated instructions 
                    for embodied virtual agents.</p>
                    <h5>PROCEDURES</h5>
                    <p>During this project, the following will happen: you will first be provided instructions, 
                    a video of a virtual agent's actions, textual descriptions of these actions, and asked a series 
                    of questions about these descriptions. The study will take approximately 5-10 minutes to 
                    complete. We recommend running the study on the latest Microsoft Edge or Google Chrome browser. 
                    Microsoft will collect the answers you provide.</p>
                    <p>Approximately 20 participants will be involved in this study.</p>
                    <h5>PERSONAL INFORMATION</h5>
                    <p><strong>Personal information we collect:</strong> During the project we will not collect personal information about you. The data 
                    collection is anonymous, and we ask you to not disclose any personal information in the survey. We do collect an identifier that is provided as a parameter in the study URL but once this identifier is disassociated from your responses we may not be to remove your data from the study without re-identifying you.</p>
                    <p><strong>How we use the data we collect:</strong> The data from the study will be used to conduct research and 
                    development into embodied virtual agents.</p>
                    <p><strong>How we store data:</strong> The data we collect will stored for a period of up to 18 months or less.</p>
                    <p>For additional information or concerns about how Microsoft handles your personal information, 
                    please see the <a href="https://privacy.microsoft.com/en-us/privacystatement">Microsoft Privacy 
                    Statement</a>. For additional information or concerns about how Microsoft handles personal information for 
                    Employees, External Staff and Candidates, please see the <a 
                    href="https://msdpn.azurewebsites.net/default?LID=62">Microsoft Global Data Protection Notice.</a></p>
                    <h5>FUTURE USE OF YOUR IDENTIFIABLE INFORMATION</h5>
                    <p>Identifiers will be removed from your identifiable private information, and after such removal, the information could be used for future research studies or distributed to another investigator for future research studies without your (or your legally authorized representative's) additional informed consent.</p>
                    <h5>BENEFITS AND RISKS</h5>
                    <p><strong>Benefits:</strong> There are no direct benefits to you that might reasonably be 
                    expected as a result of being in this study. The research team expects to better understand how 
                    to design systems that teach or explain the actions of an embodied virtual agent. You will receive 
                    any public benefit that may come these Research Results being shared with the greater scientific community.</p>
                    <p><strong>Risks:</strong> There are no anticipated, foreseeable risks or discomforts to you as a result of being in this study.</p>
                    <h5>PAYMENT FOR PARTICIPATION</h5>
                    <p>You will not be paid to take part in this study.</p>
                    <h5>CONTACT INFORMATION</h5>
                    <p>Should you have any questions concerning this project, or if you are injured as a result of 
                    being in this study, please contact Alex Polozov, at <a href="mailto:polozov@microsoft.com">polozov@microsoft.com</a>. Should you have any 
                    questions about your rights as a research subject, please contact Microsoft Research Ethics 
                    Program Feedback at <a href="mailto:MSRStudyfeedback@microsoft.com">MSRStudyfeedback@microsoft.com</a>.</p>
                    <h5>CONSENT</h5>
                    <p>By clicking "I agree" below, you confirm that the study was explained to you, you had a chance to ask questions before beginning the study, and all your questions were answered satisfactorily. By clicking "I agree" below, you voluntarily consent to participate, and you do not give up any legal rights you have as a study participant. You may request a link to download this form. On behalf of Microsoft, we thank you for your contribution and look forward to your research session.</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-dismiss="modal">I agree</button>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="footer">
        <div class="container">
            <i class="fa fa-info-circle" aria-hidden="true"></i>
            <a target="_blank" href="http://go.microsoft.com/fwlink/?LinkId=518021">Microsoft Internal Data Privacy Notice</a>
            &nbsp; | &nbsp;
            <a target="_blank" href="https://www.microsoft.com/en-us/research/lab/microsoft-research-ai/">Microsoft Research AI</a>
            &nbsp; | &nbsp;
            &copy; Microsoft 2020
        </div>
    </footer>
    
    
    <script>
    function download(filename, text) {
        let pom = document.createElement('a');
        pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
        pom.setAttribute('download', filename);
    
        if (document.createEvent) {
            let event = document.createEvent('MouseEvents');
            event.initEvent('click', true, true);
            pom.dispatchEvent(event);
        }
        else {
            pom.click();
        }
    }
    $$(function() {
        $$('#likert').submit(function (e) {
            let form = $$(this);
            if (form[0].checkValidity() !== false) {
                let q1 = $$('input[name=q1Radio]:checked').val();
                let q2 = $$('input[name=q2Radio]:checked').val();
                let q3 = $$('input[name=q3Radio]:checked').val();
                let q4 = $$('input[name=q4Radio]:checked').val();
                download('$title.json', `{ "id": "$title", "q1": $${q1}, "q2": $${q2}, "q3": $${q3}, "q4": $${q4} }`);
            }
            e.preventDefault();
            e.stopPropagation();
            form.addClass('was-validated');
        });
        $$('#consentModal').modal('show');
    });
    </script>
</body>
</html>
""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='instructions_20200817.json',
                        help='Path to the JSON file with instructions')
    parser.add_argument('-o', '--out', default='html',
                        help='Subdirectory for output HTML files')
    args = parser.parse_args()

    j = json.load(open(args.input))
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    for entry in tqdm(j):
        id: str = entry["id"].replace('baseline', 'b').replace('explainer', 'e')
        instructions = '\t'.join(f"<li>{s}</li>" for s in entry["generation"])
        html = template.substitute(title=id, video=entry["video_url"], instructions=instructions)
        with open(out_dir / f"{id}.html", 'w') as out:
            out.write(html)

from transformers import  BertModel, BertTokenizer
import torch
# import torch_directml

class SimcseModel():
    def __init__(self, model_name="princeton-nlp/sup-simcse-bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        # self.dml = torch_directml.device()
        # self.model.to(self.dml)
        self.model.eval()

    def __call__(self, texts):
        """
        texts.shape = (num samples, length of )
        """
        # sentence to token ids
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # .to(self.dml)
        
        # token ids to embeddings
        with torch.no_grad():
            embeddings = self.model(**inputs)[0]
        output = embeddings.detach().cpu().numpy()
        
        return output[:,0,:]
    
    
    
    
    
    
    
if __name__ == "__main__":
    text = "Saji Varghese is a senior techno/functional ERP specialist/Business Intelligence-ETL Architect/ Managing consultant with over Seventeen (17) years of hands-on technical experience in USA implementing multiple ERP/Business Intelligence projects.  Saji has worked with Oracle ERP implementations (11i, R12) Financial  Modules, Procurement, Accounting, FAH, Manufacturing modules, Supply Chain, Human Resources (HR), Projects, Grants, Treasury, Transport module (OTM), ERP Custom Development,  Oracle Fusion Middleware, SOA platform, JDeveloper web service development, Integration of multi Org/Multi Currencies within EBS, ERP release upgrades and Oracle ERP Expert witness consulting.  Saji is also an expert in Business Intelligence Implementations using OBIEE and Discoverer with Data Modeling experience (ERWIN, SQL Modeler).  ERP Applications Experience Saji has been an ERP-BI Techno Functional Lead Managing Consultant for over several years, working in several projects, implementing ERP/Custom Solutions, Business intelligence applications and Data Warehouse solutions. Being part of the assessment of business processes and improving business process software packages according to the client's requirement, Saji had designed developed and successfully implemented Software Designs and Technical Design Documents using standard methodologies. Saji has in depth experience in converting data from legacy to ERP systems, developing interfaces between financial subsystems and implementing performance benchmarks. Saji's expertise in ERP modules include General Ledger (GL), Accounts Payable (AP), Oracle Purchasing (PO), iProcurement, iExpense, Accounts Receivable (AR), Trading Community Architecture(TCA), Fixed Assets (FA), Cash Management, Project Accounting (PA), Oracle Service Contracts, Grants, Treasury, Workflow, E-Commerce Gateway (EDI), Inventory, Order Management (OM), Property Management (PR), Human Resources (HR), Advanced Benefits (OAB), Oracle Training Administration (OTA), Oracle SCM, Oracle Advanced Pricing, Demantra, Lease Management,  iProcurement, Self Service, Business Intelligence and ERP System Administration.  Business Intelligence Development & Data Warehouse Designing and developing Enterprise BI Applications, SOA architecture solutions with OBIEE 11G, security layer definitions, SOA BPEL integrations with Oracle ERP, PeopleSoft ERP, BI Apps, DAC, Informatica (8.X, 9.X), OBIEE (10G,11G), Answers, Dashboards, BI Publisher/XML Publisher, SOA platform, Discoverer (Discoverer Plus, Administration, User Management), Data Analysis, Business/Functional Requirement Gathering, Strategies for ETL (Sourcing, Mapping,  Process Flow), converting user requirements to ETL process, working with Oracle Warehouse Builder(OWB) to manage the entire Oracle 11i/Oracle R12 modules (Financial + Manufacturing), ODI for OWB migration and new developments"
    model = SimcseModel()
    print(model(text))

